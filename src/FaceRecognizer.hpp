#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <filesystem>

namespace fs = std::filesystem;

class FaceRecognizer {
public:
    FaceRecognizer();

    // Trains the face recognition model using images from the specified folder.
    // Returns true on success, false on failure.
    bool train(const std::string& data_folder, const std::string& model_path, const std::string& label_map_path);

    // Loads the model and label map from specified paths.
    // Returns true on success, false on failure.
    bool load(const std::string& model_path, const std::string& label_map_path);

    // Predicts the name for a given face ROI.
    // Takes a grayscale face region of interest.
    // Returns the predicted name ("Unknown" if not recognized or error).
    std::string predict(const cv::Mat& face_roi) const;

    // Checks if the model is loaded, trained, and ready for prediction.
    bool is_ready() const;

    // Gets the current label-to-name map (e.g., for GUI checks).
    const std::map<std::string, int>& get_label_map() const;

private:
    void save_label_map(const std::string& path) const;
    bool load_label_map(const std::string& path);

    // --- Configuration ---
    cv::Size training_img_size_{100, 100}; // Size images are resized to for training/prediction
    // double confidence_threshold_ = 90.0; // Example LBPH confidence threshold (lower is better)

    // --- Model Data ---
    cv::Ptr<cv::face::LBPHFaceRecognizer> model_;
    std::map<std::string, int> label_to_name_map_;
    bool is_trained_ = false;
}; 