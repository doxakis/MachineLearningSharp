using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tesseract;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace MachineLearningSharp
{
    public static class ImageToText
    {
        public static string GetTextFromPage(string filePath)
        {
            var text = new StringBuilder();
            var subImgs = GetTextAreas(filePath);
            foreach (var im in subImgs)
            {
                using (var engine = new TesseractEngine(@"./tessdata", "eng"))
                {
                    engine.DefaultPageSegMode = PageSegMode.SingleLine;
                    text.Append(engine.Process(im).GetText());
                }
            }

            return text.ToString();
        }

        public static List<Pix> GetTextAreas(string filePath)
        {
            using (Mat src = new Mat(filePath, ImreadModes.Color))
            using (Mat gray = new Mat())
            {
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

                MSER mser = MSER.Create();
                Point[][] msers = null;
                OpenCvSharp.Rect[] boundingBoxes = null;
                mser.DetectRegions(gray, out msers, out boundingBoxes); // MSER::operator()

                int meanWidth = (int)(boundingBoxes.Select(x => x.Width).Average());
                int stdWidth = (int)(
                    Math.Sqrt(
                        boundingBoxes.Select(
                            x => x.Width
                        ).Average(
                            x => x * x
                        ) - Math.Pow(meanWidth, 2)));

                int meanHeight = (int)(boundingBoxes.Select(x => x.Height).Average());
                int stdHeight = (int)(
                    Math.Sqrt(
                        boundingBoxes.Select(
                            x => x.Height
                        ).Average(
                            x => x * x
                        ) - Math.Pow(meanHeight, 2)));

                foreach (OpenCvSharp.Rect rect in boundingBoxes)
                {
                    rect.Inflate(6, 2);
                    if(rect.Width - meanWidth < stdWidth && rect.Height - meanHeight < stdHeight)
                        gray.Rectangle(rect, Scalar.Black, -1);
                }
                var bitmapToPix = new BitmapToPixConverter();
                return ExtractTextAreasFromMask(gray).Select(x => bitmapToPix.Convert(x.ToBitmap())).ToList();
            }
        }

        public static List<Mat> ExtractTextAreasFromMask(Mat image)
        {
            Mat thresh = new Mat();
            Cv2.Threshold(image, thresh, 1, 255, ThresholdTypes.Binary);
            MatOfPoint[] mops = Cv2.FindContoursAsMat(thresh, RetrievalModes.List, ContourApproximationModes.ApproxSimple);//.Select(x => (Mat)x).Where(x => x.Width > 40 && x.Height > 10).ToList();
            var matList = new List<Mat>();
            foreach (var mop in mops)
            {
                var maxX = mop.Max(x => x.X);
                var maxY = mop.Max(x => x.Y);
                var minX = mop.Min(x => x.X);
                var minY = mop.Min(x => x.Y);
                if (maxY > minY && maxX > minX)
                    matList.Add(image.SubMat(minY, maxY, minX, maxX));
            }
            return matList.Where(x => x.Width > 20 && x.Height > 5).ToList();
        }
    }
}
