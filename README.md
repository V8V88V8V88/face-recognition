# Face Recognition GUI Project

A C++ application using GTKmm and OpenCV for real-time face capture, training, and recognition. Also includes a standalone Python script for face detection.

## Overview

This project provides:
1.  A graphical user interface (GUI) built with GTKmm to:
    *   Capture face images using a webcam.
    *   Train a K-Nearest Neighbors (KNN) model based on captured faces.
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
│   ├── knn_model.yml       # Trained KNN recognition model (generated)
│   └── label_mapping.txt   # Maps internal labels to names (generated)
├── src/                    # Source code
│   ├── main_gui.cpp        # C++ Main application logic and GUI
│   ├── FaceRecognizer.cpp  # C++ Face recognition class implementation
│   ├── FaceRecognizer.hpp  # C++ Face recognition class header
│   └── facedetection.py    # Standalone Python face detection script
├── include/                # Include directory (if used for headers)
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

## Installation & Building (C++ GUI Application)

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

3.  **Create Build Directory:**
    ```bash
    mkdir build
    cd build
    ```

4.  **Run CMake:**
    ```bash
    cmake ..
    ```

5.  **Compile:**
    ```bash
    make
    ```
    An executable named `FaceRecognitionGui` will be created in the `build/bin/` directory.

## Running the C++ GUI Application

1.  **Ensure Models Directory Exists:** The application expects the `models/` directory to exist in the project root. The `haarcascade_frontalface_alt2.xml` file must be present inside it. You can usually find this file within your OpenCV installation's data directory or download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades). Copy it to the `models/` directory.
    ```bash
    # Navigate to project root first if not already there
    cd /path/to/face-recognition
    mkdir -p models
    # Example: Copying from a typical Fedora OpenCV install path
    # Adjust the source path based on your system and OpenCV installation
    cp /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml models/
    # Or download and place it there
    ```

2.  **Navigate to the build directory:**
    ```bash
    cd /path/to/face-recognition/build
    ```

3.  **Run the executable:**
    ```bash
    ./bin/FaceRecognitionGui
    ```

## Usage Guide (C++ GUI Application)

1.  **Capture:**
    *   Enter the name of the person in the "Enter Name" field.
    *   Click the "Capture" button repeatedly (10 times by default) while the person's face is clearly visible in the video feed.
    *   Captured images are saved as grayscale JPGs in the `data/faces/` directory.
2.  **Train:**
    *   Once you have captured images for one or more people, click the "Train Model" button.
    *   This will process the images in `data/faces/` and create/update the `knn_model.yml` and `label_mapping.txt` files in the `models/` directory.
3.  **Detect:**
    *   After training, click "Start Detection".
    *   The application will detect faces in the video feed and attempt to recognize them based on the trained model. Recognized faces will be shown with a green bounding box and name; unknown faces will have a red box.
    *   Click "Stop Detection" to halt the recognition process.

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

## License

Specify your license here (e.g., MIT License).
