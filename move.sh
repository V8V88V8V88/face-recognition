#!/bin/bash

# Create the directory structure
mkdir -p face-recognition/{src,models,data/faces,tests,logs,bin}

# Move all .cpp and .py files to the src directory
mv *.cpp face-recognition/src/ 2>/dev/null || echo "No .cpp files found."
mv *.py face-recognition/src/ 2>/dev/null || echo "No .py files found."
mv FaceDetectionLite.exe face-recognition/src/ 2>/dev/null || echo "FaceDetectionLite.exe not found."

# Move binary files (executables) to the bin directory
mv FaceRecognition face-recognition/bin/ 2>/dev/null || echo "FaceRecognition binary not found."
mv FaceRecognition2 face-recognition/bin/ 2>/dev/null || echo "FaceRecognition2 binary not found."
mv FaceRecognition3 face-recognition/bin/ 2>/dev/null || echo "FaceRecognition3 binary not found."
mv FaceDetectionLite face-recognition/bin/ 2>/dev/null || echo "FaceDetectionLite binary not found."

# Move model-related files to the models directory
mv deploy.prototxt.txt face-recognition/models/ 2>/dev/null || echo "deploy.prototxt.txt not found."
mv res10_300x300_ssd_iter_140000.caffemodel face-recognition/models/ 2>/dev/null || echo "res10_300x300_ssd_iter_140000.caffemodel not found."
mv dlib_face_recognition_resnet_model_v1.dat face-recognition/models/ 2>/dev/null || echo "dlib_face_recognition_resnet_model_v1.dat not found."
mv shape_predictor_5_face_landmarks.dat face-recognition/models/ 2>/dev/null || echo "shape_predictor_5_face_landmarks.dat not found."
mv shape_predictor_68_face_landmarks.dat face-recognition/models/ 2>/dev/null || echo "shape_predictor_68_face_landmarks.dat not found."
mv lbpcascade_frontalface.xml face-recognition/models/ 2>/dev/null || echo "lbpcascade_frontalface.xml not found."

# Move image files to the faces directory under data
mv faces/*.jpg face-recognition/data/faces/ 2>/dev/null || echo "No .jpg files found in faces directory."

# Move credentials and README to the data directory
mv credentials.csv face-recognition/data/ 2>/dev/null || echo "credentials.csv not found."
mv README.md face-recognition/data/ 2>/dev/null || echo "README.md not found."

# Move test files to the tests directory
mv test.png face-recognition/tests/ 2>/dev/null || echo "test.png not found."
mv test.mp4 face-recognition/tests/ 2>/dev/null || echo "test.mp4 not found."
mv simple.cpp face-recognition/tests/ 2>/dev/null || echo "simple.cpp not found."

# Move log and package files to the logs directory
mv package-lock.json face-recognition/logs/ 2>/dev/null || echo "package-lock.json not found."

# Print message on completion
echo "Files organized successfully!"
