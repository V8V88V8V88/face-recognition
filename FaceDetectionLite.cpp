#include <opencv2/opencv.hpp> // OpenCV library for computer vision
#include <opencv2/dnn.hpp>    // OpenCV deep neural networks module
#include <iostream>           // Standard input-output library

using namespace cv;            // OpenCV namespace
using namespace cv::dnn;       // OpenCV DNN namespace
using namespace std;           // Standard C++ namespace

// Path to the pre-trained model configuration and binary files
const string MODEL_CONFIG = "deploy.prototxt.txt";
const string MODEL_BINARY = "res10_300x300_ssd_iter_140000.caffemodel";

// Function to detect faces in live video
void detectFaces() {
    // Load the pre-trained model for face detection
    Net net = readNetFromCaffe(MODEL_CONFIG, MODEL_BINARY);

    // Access the default camera for video capture
    VideoCapture video(0);
    if (!video.isOpened()) {  // Check if the camera couldn't be opened
        cerr << "Error: Couldn't open the camera." << endl;
        return;
    }

    Mat frame; // Variable to store video frames

    while (true) { // Continuously process video frames
        video >> frame; // Capture a frame from the camera
        if (frame.empty()) { // Check if the frame is empty
            cerr << "Error: Couldn't capture frame." << endl;
            break;
        }

        // Prepare the frame for neural network input
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
        net.setInput(blob); // Set the frame as input to the neural network
        Mat detections = net.forward(); // Perform face detection using the neural network

        // Iterate through detected faces and draw rectangles around high-confidence faces (Highly technical stuff) DO NOT TOUCH!!!!!!!!!
        for (int i = 0; i < detections.size[2]; ++i) {
            float* dataPtr = detections.ptr<float>(0);
            float detectionConfidence = dataPtr[i * 7 + 2];

            if (detectionConfidence > 0.5) { // Check face detection confidence threshold
                // Extract coordinates for drawing the rectangle around the face of the person
                int x1 = static_cast<int>(dataPtr[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(dataPtr[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(dataPtr[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(dataPtr[i * 7 + 6] * frame.rows);

                Rect faceRect(x1, y1, x2 - x1, y2 - y1); // Rectangle around the face cordinates

                rectangle(frame, faceRect, Scalar(0, 255, 0), 2); // Draw the rectangle on the frame
            }
        }

        imshow("Face Detection", frame); // Display the processed frame

        if (waitKey(30) == 27) { // Check for 'Esc' key press to exit the program
            cout << "Exiting program." << endl;
            break;
        }
    }
}

// Main function
int main() {
    detectFaces(); // Call the function to detect faces in live video
    return 0;
}
