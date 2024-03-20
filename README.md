# Face Recognition Using C++

This repository contains a C++-based program for face detection and recognition using OpenCV and the dlib library. The program utilizes the dlib frontal face detector and shape predictor for face detection, and implements face recognition using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm.

## Features

- Detects faces in real-time using a webcam or video feed
- Recognizes known faces based on pre-loaded images and labels them with confidence

## Requirements

- OpenCV
- dlib
- `shape_predictor_68_face_landmarks.dat` file (for face detection)

## Usage

1. Ensure the required libraries are installed.
2. Place the `shape_predictor_68_face_landmarks.dat` file in the project directory.
3. Replace the placeholder face images (`known_face_1.jpg`, `known_face_2.jpg`) with your own known face images.
4. Build and run the C++ program.

## Steps to Use

### 1. Clone the Repository

```bash
git clone https://github.com/V8V88V8V88/face-recognition.git
cd face-recognition
