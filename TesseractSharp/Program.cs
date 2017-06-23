using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tesseract;

namespace TesseractSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Ensure you have Visual Studio 2015 x86 & x64 runtimes installed :
            // https://www.microsoft.com/en-us/download/details.aspx?id=48145

            var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            var testImagePath = projectDir + @"\samples\sample3.png";

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            int count = 10;
            Parallel.For(0, count, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, (index) =>
            {
                RunTesseract(projectDir, testImagePath);
            });

            stopwatch.Stop();

            Console.WriteLine("Duration: " + stopwatch.Elapsed.ToString());
            Console.WriteLine("Average speed (ms/page): " + (stopwatch.ElapsedMilliseconds / count));

            Console.Write("Press any key to continue . . . ");
            Console.ReadKey();
        }

        private static void RunTesseract(string projectDir, string testImagePath)
        {
            try
            {
                using (var engine = new TesseractEngine(projectDir + @"\tessdata", "eng", EngineMode.Default))
                {
                    using (var img = Pix.LoadFromFile(testImagePath))
                    {
                        using (var page = engine.Process(img))
                        {
                            var text = page.GetText();
                            Console.WriteLine("Mean confidence: {0}", page.GetMeanConfidence());

                            Console.WriteLine("Text (GetText): \r\n{0}", text);
                            Console.WriteLine("Text (iterator):");
                            using (var iter = page.GetIterator())
                            {
                                iter.Begin();

                                do
                                {
                                    do
                                    {
                                        do
                                        {
                                            do
                                            {
                                                if (iter.IsAtBeginningOf(PageIteratorLevel.Block))
                                                {
                                                    Console.WriteLine("<BLOCK>");
                                                }

                                                Console.Write(iter.GetText(PageIteratorLevel.Word));
                                                Console.Write(" ");

                                                if (iter.IsAtFinalOf(PageIteratorLevel.TextLine, PageIteratorLevel.Word))
                                                {
                                                    Console.WriteLine();
                                                }
                                            } while (iter.Next(PageIteratorLevel.TextLine, PageIteratorLevel.Word));

                                            if (iter.IsAtFinalOf(PageIteratorLevel.Para, PageIteratorLevel.TextLine))
                                            {
                                                Console.WriteLine();
                                            }
                                        } while (iter.Next(PageIteratorLevel.Para, PageIteratorLevel.TextLine));
                                    } while (iter.Next(PageIteratorLevel.Block, PageIteratorLevel.Para));
                                } while (iter.Next(PageIteratorLevel.Block));
                            }
                        }
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Unexpected Error: " + e.Message);
                Console.WriteLine("Details: ");
                Console.WriteLine(e.ToString());
            }
        }
    }
}
