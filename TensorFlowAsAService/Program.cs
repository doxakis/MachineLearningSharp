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
			for (int i = 0; i < 10; i++)
			{
				var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
				var modelInceptionV3 = Path.Combine(projectDir, "inception_v3.pb");
				var labelsFile = Path.Combine(projectDir, "imagenet_comp_graph_label_strings.txt");

				var modelContent = File.ReadAllBytes(modelInceptionV3);
				var labels = File.ReadAllLines(labelsFile);
				var srcDir = projectDir + @"\samples";

				var files = Directory.EnumerateFiles(srcDir, "*", SearchOption.AllDirectories)
					.Select(m => m.ToLower())
					.Where(m => m.EndsWith(".png") || m.EndsWith(".jpeg") || m.EndsWith(".jpg") || m.EndsWith(".bmp"))
					.OrderBy(m => Path.GetFileNameWithoutExtension(m))
					.ToList();

				Stopwatch stopwatch = new Stopwatch();
				stopwatch.Start();

				// One line to get the prediction from tensorflow. (model: InceptionV3 only)
				// - App must run on 64 bit.
				// - Model and jpeg can be read from file/database/etc.
				// - Labels is the classification labels.

				List<FileClassificationResult> result = TensorFlowClassificationWrapper.FindLabel(modelContent, labels, files);

				// Show result.

				var labelForOCR = new string[]
				{
					"menu",
					"web site",
					"book jacket",
					"envelope",
					"packet",
					"letter opener",
					"scoreboard",
					"binder"
				};

				foreach (var fileResult in result)
				{
					// Predictions.
					var predictions = fileResult.Predictions.OrderByDescending(m => m.Percent)
						.Where(m => m.Percent >= 0.10)
						.Take(5);

					// Use OCR ?
					if (predictions.Any(m => labelForOCR.Contains(m.Label)))
					{
						// File.
						Console.WriteLine("You should run OCR on " + Path.GetFileNameWithoutExtension(fileResult.File));

						//foreach (var prediction in predictions)
						//{
						//    Console.WriteLine(" = " + prediction.Label + " (" + prediction.Percent + ")");
						//}
					}
					else
					{
						Console.WriteLine("Look like there is no text on " + Path.GetFileNameWithoutExtension(fileResult.File));

						//foreach (var prediction in predictions)
						//{
						//    Console.WriteLine(" = " + prediction.Label + " (" + prediction.Percent + ")");
						//}
					}
				}

				stopwatch.Stop();

				Console.WriteLine();
				Console.WriteLine("Duration: " + stopwatch.Elapsed.ToString());
				Console.WriteLine("Files: " + files.Count);
				Console.WriteLine("Processor count: " + Environment.ProcessorCount);

				Console.WriteLine("Milliseconds per picture: " + (stopwatch.ElapsedMilliseconds / files.Count));

				Console.WriteLine("Press any key to continue...");
				Console.ReadLine();
			}
        }
    }
}
