#!/bin/bash

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Face Recognition Setup Script =====${NC}"

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p models
mkdir -p data/faces

# Download OpenCV models
echo -e "${BLUE}Downloading face detection models...${NC}"

# Face Detection Models
echo -e "  ${YELLOW}Downloading Haar Cascade frontal face detector...${NC}"
curl -L -o models/haarcascade_frontalface_alt2.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

echo -e "  ${YELLOW}Downloading Haar Cascade eye detector (optional)...${NC}"
curl -L -o models/haarcascade_eye.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml

# Check if CMake is installed
echo -e "${BLUE}Checking for CMake...${NC}"
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}CMake not found. You need to install CMake to build the project.${NC}"
    echo -e "  On Ubuntu/Debian: ${YELLOW}sudo apt install cmake${NC}"
    echo -e "  On Fedora/CentOS: ${YELLOW}sudo dnf install cmake${NC}"
    echo -e "  On macOS with Homebrew: ${YELLOW}brew install cmake${NC}"
else
    echo -e "${GREEN}CMake found: $(cmake --version | head -n 1)${NC}"
fi

# Check for OpenCV
echo -e "${BLUE}Checking for OpenCV...${NC}"
if pkg-config --exists opencv4; then
    echo -e "${GREEN}OpenCV found: $(pkg-config --modversion opencv4)${NC}"
else
    echo -e "${YELLOW}OpenCV not found via pkg-config. You may need to install it:${NC}"
    echo -e "  On Ubuntu/Debian: ${YELLOW}sudo apt install libopencv-dev${NC}"
    echo -e "  On Fedora/CentOS: ${YELLOW}sudo dnf install opencv-devel${NC}"
    echo -e "  On macOS with Homebrew: ${YELLOW}brew install opencv${NC}"
fi

# Check for GTKmm
echo -e "${BLUE}Checking for GTKmm 4.0...${NC}"
if pkg-config --exists gtkmm-4.0; then
    echo -e "${GREEN}GTKmm 4.0 found: $(pkg-config --modversion gtkmm-4.0)${NC}"
else
    echo -e "${YELLOW}GTKmm 4.0 not found. You need to install it:${NC}"
    echo -e "  On Ubuntu/Debian: ${YELLOW}sudo apt install libgtkmm-4.0-dev${NC}"
    echo -e "  On Fedora/CentOS: ${YELLOW}sudo dnf install gtkmm4.0-devel${NC}"
    echo -e "  On macOS with Homebrew: ${YELLOW}brew install gtkmm4${NC}"
fi

# Verify downloads
echo -e "${BLUE}Verifying downloads...${NC}"
FILES=(
    "models/haarcascade_frontalface_alt2.xml"
    "models/haarcascade_eye.xml"
)

all_files_exist=true
for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: $file failed to download${NC}"
        all_files_exist=false
    else
        echo -e "  ${GREEN}✓ $file${NC}"
    fi
done

# Setup build directory
echo -e "${BLUE}Setting up build directory...${NC}"
mkdir -p build
cd build
cmake .. || { echo -e "${RED}CMake configuration failed${NC}"; exit 1; }
echo -e "${GREEN}CMake configuration successful${NC}"

if [ "$all_files_exist" = true ]; then
    echo -e "${GREEN}All models downloaded successfully!${NC}"
    echo -e "${BLUE}===== Setup complete =====${NC}"
    echo -e "You can now build the application by running:"
    echo -e "  ${YELLOW}cd build && make${NC}"
    echo -e "And then run it with:"
    echo -e "  ${YELLOW}./bin/FaceRecognitionGui${NC}"
else
    echo -e "${RED}Setup partially complete: Some model files are missing.${NC}"
    echo -e "You can try downloading them manually from:"
    echo -e "  ${BLUE}https://github.com/opencv/opencv/tree/master/data/haarcascades${NC}"
    exit 1
fi 