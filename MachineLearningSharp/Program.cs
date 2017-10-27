using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using System.IO.Compression;
using System.Diagnostics;

namespace MachineLearningSharp
{
    class Program
    {
        public static void Main(string[] args)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            Console.WriteLine(ImageToText.GetTextFromPage("./samples/letter.png"));
            Console.ReadKey();

            var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;

            var files = Directory.EnumerateFiles(projectDir + @"\samples", "*", SearchOption.AllDirectories)
                .ToList();

            var modelFile = Path.Combine(projectDir, "inception_v3.pb");
            var labelsFile = Path.Combine(projectDir, "imagenet_comp_graph_label_strings.txt");

            // Load the serialized GraphDef from a file.
            var model = File.ReadAllBytes(modelFile);

            // Construct an in-memory graph from the serialized form.
            var graph = new TFGraph();
            
            GC.Collect();
            graph.Import(model, "");

            using (var session = new TFSession(graph))
            {
                var labels = File.ReadAllLines(labelsFile);
                foreach (var file in files)
                {
                    // Run inference on the image files
                    // For multiple images, session.Run() can be called in a loop (and
                    // concurrently). Alternatively, images can be batched since the model
                    // accepts batches of image data as input.
                    var tensor = CreateTensorFromImageFile(file);
                    var runner = session.GetRunner();

                    TFOutput classificationLayer = graph["softmax"][0];
                    TFOutput bottleneckLayer = graph["pool_3"][0];

                    TFOutput tIn = graph["DecodeJpeg"][0];
                    runner.AddInput(tIn, tensor).Fetch(classificationLayer, bottleneckLayer);
                    var output = runner.Run();

                    // Bottleneck result.
                    var result = output[1];
                    var values = ((Single[][][][])result.GetValue(jagged: true))[0][0][0];

                    Console.Write("Bottleneck: ");
                    for (int i = 0; i < values.Length; i++)
                    {
                        Console.Write(values[i] + " ");
                    }
                    Console.WriteLine();

                    // Classification result.

                    // output[0].Value() is a vector containing probabilities of
                    // labels for each image in the "batch". The batch size was 1.
                    // Find the most probably label index.
                    result = output[0];
                    var rshape = result.Shape;
                    if (result.NumDims != 2 || rshape[0] != 1)
                    {
                        var shape = "";
                        foreach (var d in rshape)
                        {
                            shape += $"{d} ";
                        }
                        shape = shape.Trim();

                        Console.WriteLine($"Error: expected to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape [{shape}]");
                        Console.ReadKey();
                        Environment.Exit(1);
                    }
                    
                    // You can get the data in two ways, as a multi-dimensional array, or arrays of arrays, 
                    // code can be nicer to read with one or the other, pick it based on how you want to process
                    // it
                    bool jagged = true;
                    var bestIdx = 0;
                    float p = 0, best = 0;
                    if (jagged)
                    {
                        var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            if (probabilities[i] > best)
                            {
                                bestIdx = i;
                                best = probabilities[i];
                            }
                        }
                    }
                    else
                    {
                        var val = (float[,])result.GetValue(jagged: false);
                        // Result is [1,N], flatten array
                        for (int i = 0; i < val.GetLength(1); i++)
                        {
                            if (val[0, i] > best)
                            {
                                bestIdx = i;
                                best = val[0, i];
                            }
                        }
                    }
                    Console.WriteLine($"{file} best match: [{bestIdx}] {best * 100.0}% {labels[bestIdx]}");
                }
            }

            stopwatch.Stop();

            Console.WriteLine();
            Console.WriteLine("Duration: " + stopwatch.Elapsed.ToString());
            Console.WriteLine("Files: " + files.Count);
            Console.WriteLine("Processor count: " + Environment.ProcessorCount);

            Console.WriteLine("Milliseconds per picture: " + (stopwatch.ElapsedMilliseconds / files.Count));
            
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        // Convert the image in filename to a Tensor suitable as input to the Inception model.
        static TFTensor CreateTensorFromImageFile(string file)
        {
            var contents = File.ReadAllBytes(file);
            // DecodeJpeg uses a scalar String-valued tensor as input.
            var tensor = TFTensor.CreateString(contents);
            TFGraph graph;
            TFOutput input, output;
            // Construct a graph to normalize the image
            ConstructGraphToNormalizeImage(out graph, out input, out output);
            // Execute that graph to normalize this one image
            using (var session = new TFSession(graph))
            {
                var normalized = session.Run(
                            inputs: new[] { input },
                            inputValues: new[] { tensor },
                            outputs: new[] { output });
                return normalized[0];
            }
        }

        // The inception model takes as input the image described by a Tensor in a very
        // specific normalized format (a particular image size, shape of the input tensor,
        // normalized pixel values etc.).
        //
        // This function constructs a graph of TensorFlow operations which takes as
        // input a JPEG-encoded string and returns a tensor suitable as input to the
        // inception model.
        static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
        {
            // Some constants specific to the pre-trained model at:
            // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
            //
            // - The model was trained after with images scaled to 224x224 pixels.
            // - The colors, represented as R, G, B in 1-byte each were converted to
            //   float using (value - Mean)/Scale.
            
            graph = new TFGraph();
            input = graph.Placeholder(TFDataType.String);
            output = graph.DecodeJpeg(contents: input, channels: 3);
        }
    }
}
