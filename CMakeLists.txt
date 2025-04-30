cmake_minimum_required(VERSION 3.10)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GTKMM_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add source files
add_executable(FaceRecognitionGui
    src/main_gui.cpp
    src/FaceDetection.cpp
)

# Link libraries
target_link_libraries(FaceRecognitionGui
    ${OpenCV_LIBS}
    ${GTKMM_LIBRARIES}
    opencv_face
)

# Add compile definitions
target_compile_definitions(FaceRecognitionGui PRIVATE
    MODELS_FOLDER="${CMAKE_SOURCE_DIR}/models/"
    DATA_FOLDER="${CMAKE_SOURCE_DIR}/data/"
)

# Print package info (optional, for debugging)
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "GTKmm Include Dirs: ${GTKMM_INCLUDE_DIRS}")
message(STATUS "GTKmm Library Dirs: ${GTKMM_LIBRARY_DIRS}")
message(STATUS "GTKmm Libraries: ${GTKMM_LIBRARIES}") 