#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p data/faces

echo "Downloading face detection model..."

# Face Detection Model (Haar Cascade)
curl -L -o models/haarcascade_frontalface_alt2.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

# Verify downloads
echo "Verifying downloads..."
FILES=(
    "models/haarcascade_frontalface_alt2.xml"
)

all_files_exist=true
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file failed to download"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = true ]; then
    echo "All models downloaded successfully!"
    echo "Setup complete. You can now build and run the application."
else
    echo "Setup failed: Some files are missing. Please check your internet connection and try again."
    exit 1
fi 