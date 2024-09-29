#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dlib/opencv.h>
#include <iostream>
#include <filesystem>
#include <unordered_map>

using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

const string faces_folder = "faces";

void recognizeFaces(const vector<Mat>& knownFaceImages, const vector<string>& knownFaceNames) {
    Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();

    // Convert string labels to integers for training
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

    CascadeClassifier faceCascade;
    if (!faceCascade.load(samples::findFile("lbpcascade_frontalface.xml"))) {
        cerr << "Error: Couldn't load face cascade." << endl;
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

        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (const auto& face : faces) {
            Mat faceROI = grayFrame(face);
            resize(faceROI, faceROI, Size(100, 100), 1.0, 1.0, INTER_CUBIC);

            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(faceROI, predictedLabel, confidence);

            string personName = "Unknown";
            for (const auto& entry : labelMapping) {
                if (entry.second == predictedLabel) {
                    personName = entry.first;
                    break;
                }
            }

            rectangle(frame, face, Scalar(0, 255, 0), 2);
            putText(frame, "Person: " + personName, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
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

    for (const auto& entry : fs::directory_iterator(faces_folder)) {
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

