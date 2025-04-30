#include "FaceRecognizer.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/face.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

FaceRecognizer::FaceRecognizer() : is_trained_(false) {
    // Create LBPH face recognizer with custom parameters for better recognition
    model_ = cv::face::LBPHFaceRecognizer::create(
        1,      // radius
        8,      // neighbors
        8,      // grid_x
        8,      // grid_y
        200.0   // threshold
    );
}

bool FaceRecognizer::train(const std::string& data_folder, const std::string& model_path, const std::string& label_map_path) {
    try {
        std::vector<cv::Mat> training_data;
        std::vector<int> labels;
        label_to_name_map_.clear();

        std::cout << "Starting training with images from: " << data_folder << std::endl;
        
        // Count files for better progress reporting
        int total_files = 0;
        for (const auto& entry : fs::directory_iterator(data_folder)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                total_files++;
            }
        }
        
        std::cout << "Found " << total_files << " image files to process" << std::endl;
        int processed = 0;

        // Process all images in the data folder
        for (const auto& entry : fs::directory_iterator(data_folder)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                // Extract name from filename (remove extension and timestamp)
                std::string name = entry.path().stem().string();
                
                // Handle different naming formats
                // Format: "name_timestamp.jpg" or just "name.jpg"
                size_t underscore_pos = name.find_last_of('_');
                if (underscore_pos != std::string::npos) {
                    // Check if what follows the underscore is a timestamp (all digits)
                    std::string potential_timestamp = name.substr(underscore_pos + 1);
                    bool is_timestamp = !potential_timestamp.empty() && 
                                       std::all_of(potential_timestamp.begin(), 
                                                  potential_timestamp.end(), 
                                                  [](unsigned char c){ return std::isdigit(c); });
                    
                    if (is_timestamp) {
                        name = name.substr(0, underscore_pos);
                    }
                }
                
                // Load image in grayscale
                cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                if (img.empty()) {
                    std::cerr << "Failed to load image: " << entry.path().string() << std::endl;
                    continue;
                }

                // Ensure image is in 8-bit format
                if (img.type() != CV_8U) {
                    img.convertTo(img, CV_8U);
                }

                // Apply histogram equalization for better feature extraction
                cv::Mat equalized;
                cv::equalizeHist(img, equalized);

                // Resize image
                cv::Mat resized;
                cv::resize(equalized, resized, training_img_size_, 0, 0, cv::INTER_LINEAR);

                // Add to training data
                training_data.push_back(resized);
                
                // Assign numeric label to name
                if (label_to_name_map_.find(name) == label_to_name_map_.end()) {
                    int label = static_cast<int>(label_to_name_map_.size());
                    label_to_name_map_[name] = label;
                    std::cout << "Assigned label " << label << " to name " << name << std::endl;
                }
                labels.push_back(label_to_name_map_[name]);
                
                // Update progress
                processed++;
                if (processed % 10 == 0 || processed == total_files) {
                    std::cout << "Processed " << processed << "/" << total_files << " images" << std::endl;
                }
            }
        }

        if (training_data.empty()) {
            std::cerr << "No valid training images found in " << data_folder << std::endl;
            return false;
        }

        std::cout << "Training with " << training_data.size() << " images for " 
                  << label_to_name_map_.size() << " people" << std::endl;

        // Verify data format
        for (size_t i = 0; i < training_data.size(); i++) {
            if (training_data[i].empty() || training_data[i].type() != CV_8U) {
                std::cerr << "Invalid image format at index " << i << std::endl;
                return false;
            }
        }

        // Train the LBPH model
        model_->train(training_data, labels);
        is_trained_ = true;

        // Save the model and label map
        model_->write(model_path);
        save_label_map(label_map_path);

        std::cout << "Training completed successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return false;
    }
}

bool FaceRecognizer::load(const std::string& model_path, const std::string& label_map_path) {
    try {
        model_.release();
        model_ = cv::face::LBPHFaceRecognizer::create();
        
        model_->read(model_path);
        
        if (!load_label_map(label_map_path)) {
            std::cerr << "Failed to load label map from " << label_map_path << std::endl;
            return false;
        }
        
        is_trained_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

std::string FaceRecognizer::predict(const cv::Mat& face_region) const {
    if (!is_trained_ || face_region.empty()) {
        return "Unknown";
    }

    try {
        // Convert to grayscale if needed
        cv::Mat gray_face;
        if (face_region.channels() > 1) {
            cv::cvtColor(face_region, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face_region.clone();
        }

        // Ensure 8-bit format
        if (gray_face.type() != CV_8U) {
            gray_face.convertTo(gray_face, CV_8U);
        }

        // Apply preprocessing to improve recognition
        cv::Mat processed_face;
        
        // 1. Apply histogram equalization to normalize lighting
        cv::equalizeHist(gray_face, processed_face);
        
        // 2. Apply slight Gaussian blur to reduce noise
        cv::GaussianBlur(processed_face, processed_face, cv::Size(3, 3), 0);

        // Resize to match training size
        cv::Mat resized_face;
        cv::resize(processed_face, resized_face, training_img_size_, 0, 0, cv::INTER_LINEAR);

        // Predict using LBPH
        int predicted_label = -1;
        double confidence = 0.0;
        model_->predict(resized_face, predicted_label, confidence);

        // Adaptive confidence threshold based on number of people trained
        // When more people are in the database, we need to be more strict
        double base_threshold = 70.0;
        const int num_people = label_to_name_map_.size();
        double adaptive_threshold = base_threshold;
        
        if (num_people > 1) {
            // Make threshold stricter with more people
            adaptive_threshold = std::max(50.0, base_threshold - (num_people * 2.0));
        }

        std::cout << "Prediction confidence: " << confidence 
                  << ", threshold: " << adaptive_threshold 
                  << " (based on " << num_people << " people)" << std::endl;
        
        // Check confidence threshold - if confidence is too high, return Unknown
        // For LBPH, lower confidence values are better (it measures distance)
        if (confidence > adaptive_threshold) {
            std::cout << "Face detected but confidence too low: " << confidence << std::endl;
            return "Unknown";
        }

        // Find the name corresponding to the predicted label
        for (const auto& pair : label_to_name_map_) {
            if (pair.second == predicted_label) {
                std::cout << "Recognized " << pair.first << " with confidence: " << confidence << std::endl;
                return pair.first;
            }
        }

        return "Unknown";
    } catch (const std::exception& e) {
        std::cerr << "Error during prediction: " << e.what() << std::endl;
        return "Unknown";
    }
}

bool FaceRecognizer::is_ready() const {
    return model_ && is_trained_ && !label_to_name_map_.empty();
}

const std::map<std::string, int>& FaceRecognizer::get_label_map() const {
    return label_to_name_map_;
}

void FaceRecognizer::save_label_map(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label map file for writing");
    }

    for (const auto& pair : label_to_name_map_) {
        file << pair.first << " " << pair.second << "\n";
    }
}

bool FaceRecognizer::load_label_map(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    label_to_name_map_.clear();
    std::string name;
    int label;
    while (file >> name >> label) {
        label_to_name_map_[name] = label;
    }

    return !label_to_name_map_.empty();
} 