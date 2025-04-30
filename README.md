# Face Recognition GUI Project

A C++ application using GTKmm and OpenCV for real-time face capture, training, and recognition. Also includes a standalone Python script for face detection.

## Overview

This project provides:
1.  A graphical user interface (GUI) built with GTKmm to:
    *   Capture face images using a webcam.
    *   Train a Local Binary Pattern Histograms (LBPH) model based on captured faces.
    *   Perform real-time face detection and recognition using the trained model and a Haar Cascade classifier.
2.  A separate Python script (`src/facedetection.py`) for performing face detection tasks (details below).

## Project Structure

```bash
face-recognition/
├── build/                  # Build directory (created by CMake for C++ app)
├── data/
│   └── faces/              # Stores captured face images (.jpg for C++ app)
├── models/                 # Required model files for C++ app
│   ├── haarcascade_frontalface_alt2.xml # Face detection model
│   ├── haarcascade_eye.xml # Optional eye detection model
│   ├── face_model.yml      # Trained LBPH recognition model (generated)
│   └── label_mapping.txt   # Maps internal labels to names (generated)
├── src/                    # Source code
│   ├── main_gui.cpp        # C++ Main application logic and GUI
│   ├── FaceRecognizer.cpp  # C++ Face recognition class implementation
│   ├── FaceRecognizer.hpp  # C++ Face recognition class header
│   └── facedetection.py    # Standalone Python face detection script
├── setup.sh                # Setup script to download models and check dependencies
├── CMakeLists.txt          # CMake build configuration for C++ app
└── README.md               # This file
```

## Requirements

**For C++ GUI Application:**
- **C++ Compiler:** Supporting C++17 (e.g., GCC, Clang)
- **CMake:** Version 3.10 or higher
- **OpenCV (C++):** Core, ImgProc, HighGUI, ObjDetect, Face, ML, VideoIO modules (tested with 4.x)
- **GTKmm:** Version 4.0 (`gtkmm-4.0`)

**For Python Script (`facedetection.py`):**
- **Python:** Version 3.x recommended
- **OpenCV (Python):** `opencv-python` package
- *(Other Python dependencies might be required - check the script's imports)*

## Quick Setup & Build

The easiest way to set up the project is using the provided setup script:

```bash
# Clone the repository
git clone <your-repository-url>
cd face-recognition

# Make setup script executable (if needed)
chmod +x setup.sh

# Run setup script
./setup.sh
```

The setup script will:
1. Download required face detection models
2. Create necessary directories
3. Check for required dependencies
4. Set up the CMake build directory

Once the setup is complete, you can build the application:

```bash
# Build the application (from project root)
cd build && make
```

## Manual Installation & Building (If setup script fails)

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd face-recognition
    ```

2.  **Install C++ Dependencies:**
    *   **Fedora/CentOS:**
        ```bash
        sudo dnf install cmake gcc-c++ opencv-devel gtkmm4.0-devel
        ```
    *   **Debian/Ubuntu:**
        ```bash
        sudo apt update
        sudo apt install build-essential cmake libopencv-dev libgtkmm-4.0-dev
        ```
    *(Package names might vary slightly depending on your distribution version)*

3.  **Download Face Detection Models:**
    ```bash
    mkdir -p models
    curl -L -o models/haarcascade_frontalface_alt2.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
    ```

4.  **Create Build Directory:**
    ```bash
    mkdir build
    cd build
    ```

5.  **Run CMake:**
    ```bash
    cmake ..
    ```

6.  **Compile:**
    ```bash
    make
    ```
    An executable named `FaceRecognitionGui` will be created in the `build/bin/` directory.

## Running the Application

```bash
# From the build directory
./bin/FaceRecognitionGui
```

## Usage Guide

1.  **Capture:**
    *   Enter the name of the person in the "Enter Name" field.
    *   Click the "Capture" button repeatedly (10 times by default) while the person's face is clearly visible in the video feed.
    *   The system will check for quality (lighting, face size) and capture only when conditions are good.
    *   Captured images are saved as grayscale JPGs in the `data/faces/` directory.
    
2.  **Train:**
    *   Once you have captured images for one or more people, click the "Train Model" button.
    *   This will process the images in `data/faces/` and create/update the `face_model.yml` and `label_mapping.txt` files in the `models/` directory.
    *   Training includes histogram equalization and other image preprocessing for better recognition quality.
    
3.  **Detect:**
    *   After training, click "Start Detection".
    *   The application will detect faces in the video feed and attempt to recognize them based on the trained model. 
    *   Recognized faces will be shown with a green bounding box and name; unknown faces will have a red box.
    *   The system uses an adaptive confidence threshold that adjusts based on the number of people in the database.
    *   Click "Stop Detection" to halt the recognition process.

## Recognition Algorithm

This application uses the Local Binary Patterns Histograms (LBPH) algorithm for face recognition, which provides:

- Robustness to lighting variations
- Good performance with small training sets (as few as 10 images per person)
- Efficient recognition without requiring GPU acceleration
- Adaptive confidence threshold to better handle unknown faces

## Troubleshooting

* **Camera Not Detected:** Ensure your webcam is properly connected and not being used by another application.
* **Poor Recognition:** Try improving lighting conditions and capture images from different angles.
* **Build Issues:** Run `./setup.sh` to check for missing dependencies.
* **Model Download Errors:** Check internet connection or manually download models from the OpenCV GitHub repository.

## Using the Python Face Detection Script

The `src/facedetection.py` script provides standalone face detection capabilities.

1.  **Install Python Dependencies:**
    ```bash
    pip install opencv-python
    # Install any other dependencies imported by the script
    ```

2.  **Run the script:**
    *(Usage may vary depending on the script's implementation. Check the script's code or command-line arguments -- if any -- for specifics. Example assuming it takes an image path as input)*
    ```bash
    python src/facedetection.py --image path/to/your/image.jpg
    ```
    *(Or it might use the webcam directly)*
    ```bash
    python src/facedetection.py
    ```

## Contributing

Contributions are welcome! Please follow standard fork-and-pull-request procedures.