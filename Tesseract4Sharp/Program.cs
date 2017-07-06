using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tesseract4Sharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // First install: http://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-setup-4.00.00dev.exe
            // (also, available on the repo)

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            var testFiles = Directory.EnumerateFiles(projectDir + @"\samples");
            Parallel.ForEach(testFiles, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, (fileName) =>
            {
                var imageFile = File.ReadAllBytes(fileName);
                var text = ParseText(imageFile, "eng", "fra");
                Console.WriteLine("File:" + fileName + "\n" + text + "\n");
            });

            stopwatch.Stop();
            Console.WriteLine("Duration: " + stopwatch.Elapsed.ToString());

            Console.WriteLine("Press any key to continue...");
            Console.ReadLine();
        }
        
        private static string ParseText(byte[] imageFile, params string[] lang)
        {
            string output = string.Empty;
            var tempOutputFile = Path.GetTempPath() + Guid.NewGuid();
            var tempImageFile = Path.GetTempFileName();

            File.AppendAllText(tempOutputFile, "-");

            try
            {
                File.WriteAllBytes(tempImageFile, imageFile);

                ProcessStartInfo info = new ProcessStartInfo();
                info.WorkingDirectory = @"C:\Program Files (x86)\Tesseract-OCR";
                info.WindowStyle = ProcessWindowStyle.Hidden;
                info.FileName = "tesseract.exe";
                info.Arguments =
                    // Image file.
                    tempImageFile + " " +
                    // Output file (tesseract add '.txt' at the end)
                    tempOutputFile +
                    // Use LSTM.
                    " --oem 1 " +
                    // Languages.
                    " -l " + string.Join("+", lang);
                
                // Start tesseract.
                Process process = Process.Start(info);
                process.WaitForExit();
                if (process.ExitCode == 0)
                {
                    // Exit code: success.
                    output = File.ReadAllText(tempOutputFile + ".txt");
                }
            }
            catch
            {
                File.Delete(tempImageFile);
                File.Delete(tempOutputFile + ".txt");
            }
            return output;
        }
    }
}
