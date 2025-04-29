cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect dnn face videoio imgcodecs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${GTKMM_INCLUDE_DIRS}
)

# Link libraries
link_directories(${GTKMM_LIBRARY_DIRS})

# Add the executable
add_executable(FaceRecognitionGui src/main_gui.cpp) # We will create main_gui.cpp later

# Link libraries to the executable
target_link_libraries(FaceRecognitionGui
    PRIVATE
    ${OpenCV_LIBS}
    ${GTKMM_LIBRARIES}
    stdc++fs # For <filesystem>
)

# Ensure the executable is placed in the bin directory relative to build
set_target_properties(FaceRecognitionGui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print package info (optional, for debugging)
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "GTKmm Include Dirs: ${GTKMM_INCLUDE_DIRS}")
message(STATUS "GTKmm Library Dirs: ${GTKMM_LIBRARY_DIRS}")
message(STATUS "GTKmm Libraries: ${GTKMM_LIBRARIES}") 