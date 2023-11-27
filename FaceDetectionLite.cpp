#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string MODEL_CONFIG = "deploy.prototxt.txt";
const string MODEL_BINARY = "res10_300x300_ssd_iter_140000.caffemodel";

void detectFaces() {
    Net net = readNetFromCaffe(MODEL_CONFIG, MODEL_BINARY);

    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Error: Couldn't open the camera." << endl;
        return;
    }

    Mat frame;

    while (true) {
        video >> frame;
        if (frame.empty()) {
            cerr << "Error: Couldn't capture frame." << endl;
            break;
        }

        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        net.setInput(blob);
        Mat detections = net.forward();

        for (int i = 0; i < detections.size[2]; ++i) {
            float* dataPtr = detections.ptr<float>(0);
            float detectionConfidence = dataPtr[i * 7 + 2];

            if (detectionConfidence > 0.5) {
                int x1 = static_cast<int>(dataPtr[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(dataPtr[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(dataPtr[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(dataPtr[i * 7 + 6] * frame.rows);

                Rect faceRect(x1, y1, x2 - x1, y2 - y1);

                rectangle(frame, faceRect, Scalar(0, 255, 0), 2);
            }
        }

        imshow("Face Detection", frame);

        if (waitKey(30) == 27) {
            cout << "Exiting program." << endl;
            break;
        }
    }
}

int main() {
    detectFaces();
    return 0;
}

