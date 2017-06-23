using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

namespace TensorFlowAsAService
{
    class Program
    {
        public static void Main(string[] args)
        {
            var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            var modelInceptionV3 = Path.Combine(projectDir, "inception_v3.pb");
            var labelsFile = Path.Combine(projectDir, "imagenet_comp_graph_label_strings.txt");

            var modelContent = File.ReadAllBytes(modelInceptionV3);
            var labels = File.ReadAllLines(labelsFile);
            var filesContent = Directory.EnumerateFiles(projectDir + @"\samples", "*", SearchOption.AllDirectories)
                .Select(m => File.ReadAllBytes(m))
                .ToList();

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            // One line to get the prediction from tensorflow. (model: InceptionV3 only)
            // - App must run on 64 bit.
            // - Model and jpeg can be read from file/database/etc.
            // - Labels is the classification labels.

            List<FileClassificationResult> result = TensorFlowClassificationWrapper.FindLabel(modelContent, labels, filesContent);

            stopwatch.Stop();

            Console.WriteLine();
            Console.WriteLine("Duration: " + stopwatch.Elapsed.ToString());
            Console.WriteLine("Files: " + filesContent.Count);
            Console.WriteLine("Processor count: " + Environment.ProcessorCount);

            Console.WriteLine("Milliseconds per picture: " + (stopwatch.ElapsedMilliseconds / filesContent.Count));

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }
    }
}
