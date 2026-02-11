# Face Recognition GUI Project

A C++ application using GTKmm and OpenCV for real-time face capture, training, and recognition. Also includes a standalone Python script using the dlib-based `face_recognition` library.

## Overview

This project provides:

1. **C++ GUI Application** (GTKmm + OpenCV):
   - Capture face images using a webcam
   - Train an LBPH model from captured faces
   - Real-time face detection (Haar Cascade) and recognition (LBPH)

2. **Python Script** (`src/facedetection.py`):
   - Standalone real-time face recognition using the `face_recognition` library
   - Uses pre-captured images from `data/faces/` as known faces
   - Multi-threaded processing for smoother video

## Project Structure

```
face-recognition/
├── data/
│   └── faces/              # Captured face images (.jpg) - created by setup
├── models/                 # Haar Cascade models (downloaded by setup)
│   ├── haarcascade_frontalface_alt2.xml
│   └── haarcascade_eye.xml
├── src/
│   ├── main_gui.cpp        # C++ GUI and main logic
│   ├── FaceRecognizer.cpp  # LBPH recognition implementation
│   ├── FaceRecognizer.hpp
│   └── facedetection.py    # Python face recognition script
├── tests/
│   ├── test.mp4
│   └── test.png
├── setup.sh                # Downloads models, runs cmake
├── CMakeLists.txt
├── requirements.txt       # Python dependencies
└── README.md
```

Generated at runtime (not in repo): `models/face_model.yml`, `models/label_mapping.txt`

## Requirements

### C++ GUI Application

- C++17 compiler (GCC, Clang)
- CMake 3.10+
- OpenCV (C++): Core, ImgProc, HighGUI, ObjDetect, Face, ML, VideoIO
- GTKmm 4.0

### Python Script

- Python 3.x
- opencv-python
- face-recognition (uses dlib)

## Setup & Build

### C++ Application

```bash
chmod +x setup.sh
./setup.sh
cd build && make
```

If setup fails due to missing dependencies:

**Fedora/CentOS:**
```bash
sudo dnf install cmake gcc-c++ opencv-devel gtkmm4.0-devel
```

**Debian/Ubuntu:**
```bash
sudo apt install build-essential cmake libopencv-dev libgtkmm-4.0-dev
```

Manual model download (if setup fails):
```bash
mkdir -p models
curl -L -o models/haarcascade_frontalface_alt2.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
```

## Running

### C++ GUI

Run from the project root (paths are relative):

```bash
./build/bin/FaceRecognitionGui
```

### Python Script

```bash
pip install opencv-python face-recognition
python src/facedetection.py
```

Uses webcam, reads known faces from `data/faces/*.jpg` (filename without extension = person name). Press `q` to quit.

On Wayland, you may need:
```bash
QT_QPA_PLATFORM=xcb python src/facedetection.py
```

## Usage (C++ GUI)

1. **Capture:** Enter a name, click Capture 10 times with your face visible. Images saved to `data/faces/`. Note: starting a new capture session clears existing images in that folder.

2. **Train:** Click "Train Model" to build the LBPH model from `data/faces/`. Creates `face_model.yml` and `label_mapping.txt`.

3. **Detect:** Click "Start Detection" to recognize faces. Green box = recognized, red = unknown.

## Algorithm

- **Face detection:** Haar Cascade (haarcascade_frontalface_alt2)
- **Face recognition:** LBPH (Local Binary Patterns Histograms) with adaptive confidence threshold

## Troubleshooting

- **Camera not detected:** Check webcam is connected and not in use elsewhere
- **Build fails:** Install OpenCV and GTKmm dev packages (see Setup)
- **Poor recognition:** Better lighting, capture from varied angles, ensure face is large enough in frame
