#ifndef FACE_RECOGNIZER_HPP
#define FACE_RECOGNIZER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem> // C++17 filesystem

namespace fs = std::filesystem;

class FaceRecognizer {
public:
    FaceRecognizer();

    // Trains the KNN model using images from the specified folder.
    // Returns true on success, false on failure.
    bool train(const std::string& data_folder, 
               const std::string& model_save_path, 
               const std::string& map_save_path);

    // Loads the KNN model and label map from specified paths.
    // Returns true on success, false on failure.
    bool load(const std::string& model_load_path, 
              const std::string& map_load_path);

    // Predicts the name for a given face ROI.
    // Takes a grayscale face region of interest.
    // Returns the predicted name ("Unknown" if not recognized or error).
    std::string predict(const cv::Mat& face_roi);

    // Checks if the model is loaded, trained, and ready for prediction.
    bool is_ready() const;

    // Gets the current label-to-name map (e.g., for GUI checks).
    const std::unordered_map<int, std::string>& get_label_map() const;


private:
    cv::Ptr<cv::ml::KNearest> knn_model_;
    std::unordered_map<int, std::string> label_to_name_map_;
    bool is_trained_ = false; // Track training status explicitly
    const cv::Size training_img_size_ = cv::Size(100, 100); // Standard size for training/prediction
    const int knn_k_ = 3; // K value for KNN
};

#endif // FACE_RECOGNIZER_HPP 