#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <unordered_map>

using namespace cv;
using namespace cv::dnn;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

const string FACES_FOLDER = "faces";
const string MODEL_CONFIG = "deploy.prototxt.txt";
const string MODEL_BINARY = "res10_300x300_ssd_iter_140000.caffemodel";

void recognizeFaces(const vector<Mat>& knownFaceImages, const vector<string>& knownFaceNames) {
    Net net = readNetFromCaffe(MODEL_CONFIG, MODEL_BINARY);

    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create(1, 8, 8, 8, 123.0);

    unordered_map<string, int> labelMapping;
    vector<int> labels;

    int labelIndex = 0;
    for (const auto& name : knownFaceNames) {
        if (labelMapping.find(name) == labelMapping.end()) {
            labelMapping[name] = labelIndex++;
        }
        labels.push_back(labelMapping[name]);
    }

    model->train(knownFaceImages, labels);

    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Error: Couldn't open the camera." << endl;
        return;
    }

    Mat frame;
    vector<Rect> faces;

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

                Mat faceROI = frame(faceRect);

                cvtColor(faceROI, faceROI, COLOR_BGR2GRAY);
                resize(faceROI, faceROI, Size(100, 100), 1.0, 1.0, INTER_CUBIC);

                int predictedLabel = -1;
                double predictionConfidence = 0.0;
                model->predict(faceROI, predictedLabel, predictionConfidence);

                string personName = "Unknown";
                if (predictionConfidence < 100.0) {
                    for (const auto& entry : labelMapping) {
                        if (entry.second == predictedLabel) {
                            personName = entry.first;
                            break;
                        }
                    }
                }

                rectangle(frame, faceRect, Scalar(0, 255, 0), 2);
                putText(frame, "Person: " + personName, Point(faceRect.x, faceRect.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }

        imshow("Face Recognition", frame);

        if (waitKey(30) == 27) {
            cout << "Exiting program." << endl;
            break;
        }
    }
}

int main() {
    vector<Mat> knownFaceImages;
    vector<string> knownFaceNames;

    for (const auto& entry : fs::directory_iterator(FACES_FOLDER)) {
        if (entry.path().extension() != ".jpg")
            continue;

        Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Error: Couldn't read image " << entry.path().string() << endl;
            continue;
        }

        string personName = entry.path().stem().string();

        knownFaceImages.push_back(image);
        knownFaceNames.push_back(personName);
    }

    if (knownFaceImages.empty()) {
        cerr << "Error: No face images found in the 'faces' folder." << endl;
        return 1;
    }

    recognizeFaces(knownFaceImages, knownFaceNames);

    return 0;
}

