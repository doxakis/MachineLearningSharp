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
        public string Label { get; internal set; }
        public float Percent { get; internal set; }
        public float[] Bottleneck { get; internal set; }
        public int ClassificationId { get; internal set; }
    }

    public class TensorFlowClassificationWrapper
    {
        public static List<FileClassificationResult> FindLabel(byte[] model, string[] labels, IEnumerable<byte[]> files)
        {
            List<FileClassificationResult> listOfFileClassificationResult = new List<FileClassificationResult>();

            var graph = new TFGraph();

            GC.Collect();
            graph.Import(model, "");
            
            using (var session = new TFSession(graph))
            {
                foreach (var file in files)
                {
                    FileClassificationResult fileResult = new FileClassificationResult();

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
                    fileResult.Bottleneck = values;

                    // Classification result.
                    result = output[0];

                    var bestIdx = 0;
                    float p = 0, best = 0;
                    var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                    for (int i = 0; i < probabilities.Length; i++)
                    {
                        if (probabilities[i] > best)
                        {
                            bestIdx = i;
                            best = probabilities[i];
                        }
                    }
                    fileResult.Label = labels[bestIdx];
                    fileResult.ClassificationId = bestIdx;
                    fileResult.Percent = best * 100.0f;

                    listOfFileClassificationResult.Add(fileResult);
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
