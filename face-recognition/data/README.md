# Face Recognition Using C, C++, Python 

This repository contains a C-based program for face detection and recognition using OpenCV and the dlib library. The program utilizes the dlib frontal face detector and shape predictor for face detection and implements face recognition using OpenCV's LBPH (Local Binary Patterns Histograms) algorithm.
Features:

    Detects faces in real-time using a webcam or video feed.
    Recognizes known faces based on pre-loaded images and labels them with confidence.

Requirements:

    OpenCV
    dlib
    shape_predictor_68_face_landmarks.dat file (for face detection)

Usage:

    Ensure the required libraries are installed.
    Place the shape_predictor_68_face_landmarks.dat file in the project directory.
    Replace the placeholder face images (known_face_1.jpg, known_face_2.jpg) with your own known face images.
    Build and run the C program.

Steps to Use:

    Clone the Repository:
    
    git clone https://github.com/V8V88V8V88/face-recognition.git
    cd face-recognition



Setup Environment:

    Install OpenCV: Follow the installation instructions for OpenCV (OpenCV Installation Guide).
    Install dlib: Install dlib using your package manager or compile it from source (dlib Installation Guide).
    Download shape_predictor_68_face_landmarks.dat: Download the pre-trained model file from the dlib website or related repositories.

Prepare Known Faces:

    Replace placeholder face images (known_face_1.jpg, known_face_2.jpg, etc.) with your known face images.
    Update the known_face_names vector in the C code with the corresponding names for these faces.

Build the Program:

    Compile the C program using your preferred C/C++ compiler. For example:
    
    g++ -o face_recognition face_recognition.cpp `pkg-config --cflags --libs opencv4`

Run the Program:

    Execute the built program:

bash

    ./face_recognition

Ensure your webcam is connected and operational.
The program will detect faces in real-time and recognize known faces based on the provided images, labeling them with confidence.

Contribution Guidelines:

    Fork the repository, make your changes, and create a pull request with detailed explanations of the proposed changes.

Feel free to contribute and improve the functionality! :D
