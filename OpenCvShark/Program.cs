using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenCvShark
{
    class Program
    {
        static void Main(string[] args)
        {
            var projectDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.FullName;
            var file = projectDir + @"\samples\sample3.png";
            
            using (Mat src = new Mat(file, ImreadModes.Color))
            using (Mat gray = new Mat())
            using (Mat dst = src.Clone())
            {
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

                CppStyleMSER(gray, dst);  // C++ style

                //using (new Window("MSER src", src))
                //using (new Window("MSER gray", gray))
                using (new Window("MSER dst", dst))
                {
                    Cv2.WaitKey();
                }
            }
        }

        /// <summary>
        /// Extracts MSER by C++-style code (cv::MSER)
        /// </summary>
        /// <param name="gray"></param>
        /// <param name="dst"></param>
        private static void CppStyleMSER(Mat gray, Mat dst)
        {
            MSER mser = MSER.Create();
            Point[][] contours;
            Rect[] bboxes;
            mser.DetectRegions(gray, out contours, out bboxes);

            foreach (var item in bboxes)
            {
                Scalar color = Scalar.Blue;
                dst.Rectangle(item, color);
            }

            /*foreach (Point[] pts in contours)
            {
                Scalar color = Scalar.RandomColor();
                foreach (Point p in pts)
                {
                    dst.Circle(p, 1, color);
                }
            }*/
        }
    }
}
