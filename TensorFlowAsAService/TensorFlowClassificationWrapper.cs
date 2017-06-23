using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlowAsAService
{
    public class FileClassificationResult
    {
        public string File { get; internal set; }
        public float[] Bottleneck { get; internal set; }
        public IEnumerable<Prediction> Predictions { get; internal set; }
    }

    public class Prediction
    {
        public string Label { get; internal set; }
        public int ClassificationId { get; internal set; }
        public float Percent { get; internal set; }
    }

    public class TensorFlowClassificationWrapper
    {
        public static List<FileClassificationResult> FindLabel(byte[] model, string[] labels, IEnumerable<string> files)
        {
            List<FileClassificationResult> listOfFileClassificationResult = new List<FileClassificationResult>();

            var graph = new TFGraph();

            GC.Collect();
            graph.Import(model, "");
            
            using (var session = new TFSession(graph))
            {
                foreach (var file in files)
                {
                    try
                    {
                        FileClassificationResult fileResult = new FileClassificationResult();

                        var tensor = CreateTensorFromImageFile(File.ReadAllBytes(file));
                        var runner = session.GetRunner();

                        TFOutput classificationLayer = graph["softmax"][0];
                        TFOutput bottleneckLayer = graph["pool_3"][0];

                        TFOutput tIn = graph["DecodeJpeg"][0];
                        runner.AddInput(tIn, tensor).Fetch(classificationLayer, bottleneckLayer);
                        var output = runner.Run();

                        // Bottleneck result.
                        var result = output[1];
                        var values = ((Single[][][][])result.GetValue(jagged: true))[0][0][0];
                        fileResult.Bottleneck = values;

                        // Classification result.
                        result = output[0];

                        var probabilities = ((float[][])result.GetValue(jagged: true))[0];

                        var predictions = new List<Prediction>();
                        for (int i = 0; i < probabilities.Length; i++)
                        {
                            predictions.Add(new Prediction
                            {
                                ClassificationId = i,
                                Label = (i >= labels.Length) ? "Unknown" : labels[i],
                                Percent = probabilities[i]
                            });
                        }
                        fileResult.Predictions = predictions;
                        fileResult.File = file;

                        listOfFileClassificationResult.Add(fileResult);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                    }
                }
            }
            return listOfFileClassificationResult;
        }

        // Convert the image in filename to a Tensor suitable as input to the Inception model.
        static TFTensor CreateTensorFromImageFile(byte[] contents)
        {
            // DecodeJpeg uses a scalar String-valued tensor as input.
            var inputTensor = TFTensor.CreateString(contents);
            TFGraph graph = new TFGraph();
            TFOutput input = graph.Placeholder(TFDataType.String);
            TFOutput output = graph.DecodeJpeg(contents: input, channels: 3);

            using (var session = new TFSession(graph))
            {
                var tensor = session.Run(
                    inputs: new[] { input },
                    inputValues: new[] { inputTensor },
                    outputs: new[] { output });
                return tensor[0];
            }
        }
    }
}
