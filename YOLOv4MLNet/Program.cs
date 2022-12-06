using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using YOLOv4MLNet.DataStructures;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace YOLOv4MLNet
{
    //https://towardsdatascience.com/yolo-v4-optimal-speed-accuracy-for-object-detection-79896ed47b50
    class Program
    {
        // model is available here:
        // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4
        const string modelPath = @"Assets\Models\mouse.onnx";

        const string imageFolder = @"Assets\Images";

        const string imageOutputFolder = @"Assets\Output";

        //static readonly string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
        static readonly string[] classesNames = new string[] { "cat", "dog", "mouse" };

        static void Main()
        {
            Directory.CreateDirectory(imageOutputFolder);
            MLContext mlContext = new();

            // model is available here:
            // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "images", imageWidth: 640, imageHeight: 640, resizing: ResizingKind.Fill)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "images", scaleImage: 1f / 255f, interleavePixelColors: false))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    //shapeDictionary: new Dictionary<string, int[]>()
                    //{
                    //   { "images", new[] { 1, 3, 640, 640 } },
                    //   { "output0", new[] { 1, 25200, 8 } },
                    //},
                    //inputColumnNames: new[]
                    //{
                    //   "images"
                    //},
                    //outputColumnNames: new[]
                    //{
                    //   "output0"
                    //},
                    modelFile: modelPath
                )
            );

            // 适应空列表以获取输入数据架构
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));

            // 创建预测引擎
            var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);

            // 保存模型
            //mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));

            // foreach (string imageName in new string[] { "kite.jpg", "kite_416.jpg", "dog_cat.jpg", "cars road.jpg", "ski.jpg", "ski2.jpg" })
            foreach (string imageName in new string[] { "微信图片_20221205163715.jpg", "微信图片_20221205163743.jpg", "微信图片_20221205163747.jpg", "微信图片_20221205163751.jpg", "微信图片_20221205163758.jpg" })
            {
                Console.WriteLine($"推理 {imageName} ...");
                var ss = Stopwatch.StartNew();
                ss.Start();
                using (var bitmap = new Bitmap(Image.FromFile(Path.Combine(imageFolder, imageName))))
                {
                    // 预测 推理
                    var predict = predictionEngine.Predict(new YoloV4BitmapData() { Image = bitmap });
                    var results = predict.GetResults(classesNames, 0.3f, 0.7f);

                    using var g = Graphics.FromImage(bitmap);
                    foreach (var res in results)
                    {
                        // 绘制预测结果
                        var x1 = res.BBox[0];
                        var y1 = res.BBox[1];
                        var x2 = res.BBox[2];
                        var y2 = res.BBox[3];
                        g.DrawRectangle(Pens.Red, x1, y1, x2 - x1, y2 - y1);
                        using (var brushes = new SolidBrush(Color.FromArgb(50, Color.Red)))
                        {
                            g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
                        }

                        g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"), new Font("Arial", 12), Brushes.Blue, new PointF(x1, y1));
                    }
                    bitmap.Save(Path.Combine(imageOutputFolder, Path.ChangeExtension(imageName, "_processed" + Path.GetExtension(imageName))));
                }
                ss.Stop();
                Console.WriteLine($"耗时：{ss.ElapsedMilliseconds} ms.");

            }
        }
    }
}
