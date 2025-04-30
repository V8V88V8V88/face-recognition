#include "FaceRecognizer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept> // For runtime_error

FaceRecognizer::FaceRecognizer() {
    // Initialize KNN model pointer but don't create until training/loading
}

// Trains the KNN model using images from the specified folder.
bool FaceRecognizer::train(const std::string& data_folder, 
                           const std::string& model_save_path, 
                           const std::string& map_save_path) {
    std::cout << "Starting KNN Training from folder: " << data_folder << std::endl;
    is_trained_ = false; // Reset training status
    
    cv::Mat training_data;
    cv::Mat training_labels;
    std::unordered_map<std::string, int> name_to_label; // Local map for this training session
    label_to_name_map_.clear(); // Clear the member map
    int next_label = 0;
    int images_processed = 0;

    try {
        if (!fs::exists(data_folder)) {
             std::cerr << "Error: Faces data folder not found: " << data_folder << std::endl;
             return false;
        }

        for (const auto& entry : fs::directory_iterator(data_folder)) {
            fs::path entry_path = entry.path();
            if (entry_path.extension() != ".jpg" && entry_path.extension() != ".png") continue;

            std::string filename = entry_path.stem().string();
            size_t pos = filename.find_last_of('_');
            if (pos == std::string::npos) {
                 std::cerr << "Warning: Skipping file with unexpected name format: " << entry_path << std::endl;
                 continue;
            }

            std::string name = filename.substr(0, pos);
            cv::Mat face_img = cv::imread(entry_path.string(), cv::IMREAD_GRAYSCALE);
            
            if (face_img.empty()) {
                std::cerr << "Warning: Could not read image: " << entry_path << std::endl;
                continue;
            }

            // --- Preprocessing --- 
            cv::Mat resized_face;
            cv::resize(face_img, resized_face, training_img_size_);
            cv::Mat flattened_face = resized_face.reshape(1, 1);
            flattened_face.convertTo(flattened_face, CV_32F);

            // --- Assign label --- 
            int label;
            if (name_to_label.find(name) == name_to_label.end()) {
                label = next_label++;
                name_to_label[name] = label;
                label_to_name_map_[label] = name; // Update member map
                std::cout << "Assigning label " << label << " to " << name << std::endl;
            } else {
                label = name_to_label[name];
            }

            training_data.push_back(flattened_face);
            training_labels.push_back(label);
            images_processed++;
        }

        if (training_data.empty() || training_labels.empty()) {
            std::cerr << "Error: No valid training images found!" << std::endl;
            return false;
        }

        training_labels.convertTo(training_labels, CV_32S);

        // --- Train KNN Model --- 
        knn_model_ = cv::ml::KNearest::create();
        knn_model_->setDefaultK(knn_k_);
        knn_model_->setIsClassifier(true);
        knn_model_->train(training_data, cv::ml::ROW_SAMPLE, training_labels);
        is_trained_ = knn_model_->isTrained();

        if (!is_trained_) {
             std::cerr << "Error: KNN model training failed." << std::endl;
             return false;
        }

        // --- Save Model and Map --- 
        knn_model_->save(model_save_path);
        std::cout << "KNN model saved to: " << model_save_path << std::endl;

        std::ofstream mapping_file(map_save_path);
        if (!mapping_file.is_open()) {
             throw std::runtime_error("Could not open label mapping file for writing: " + map_save_path);
        }
        for (const auto& pair : label_to_name_map_) {
            mapping_file << pair.first << "," << pair.second << "\n";
        }
        mapping_file.close();
        std::cout << "Label map saved to: " << map_save_path << std::endl;

        std::cout << "KNN Training complete: " << images_processed << " images, " 
                  << next_label << " people." << std::endl;
        return true;
        
    } catch (const cv::Exception& cv_e) {
        std::cerr << "OpenCV Error during training: " << cv_e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error during training: " << e.what() << std::endl;
    }
    return false; // Return false if any exception occurred
}

// Loads the KNN model and label map.
bool FaceRecognizer::load(const std::string& model_load_path, 
                          const std::string& map_load_path) {
    is_trained_ = false;
    knn_model_.release(); // Clear existing model
    label_to_name_map_.clear();

    try {
        // Load KNN Model
        if (fs::exists(model_load_path)) {
             knn_model_ = cv::ml::KNearest::load(model_load_path);
             if (!knn_model_ || !knn_model_->isTrained()) {
                  std::cerr << "Warning: Could not load or invalid KNN model found at " << model_load_path << std::endl;
                  knn_model_.release();
             } else {
                   std::cout << "Loaded existing KNN model from: " << model_load_path << std::endl;
                   is_trained_ = true;
             }
        } else {
             std::cout << "KNN model file not found: " << model_load_path << std::endl;
        }

        // Load Label Map
        if (fs::exists(map_load_path)) {
            std::ifstream mapping_file(map_load_path);
            if (!mapping_file.is_open()) {
                 std::cerr << "Warning: Could not open label mapping file: " << map_load_path << std::endl;
            } else {
                std::string line;
                int count = 0;
                while (std::getline(mapping_file, line)) {
                    std::istringstream iss(line);
                    std::string label_str, name;
                    if (std::getline(iss, label_str, ',') && std::getline(iss, name)) {
                        try {
                             label_to_name_map_[std::stoi(label_str)] = name;
                             count++;
                        } catch (...) { // Catch potential stoi errors
                              std::cerr << "Warning: Invalid format in mapping file line: " << line << std::endl;
                        }
                    }
                }
                mapping_file.close();
                 if (count > 0) {
                      std::cout << "Loaded map with " << count << " people from: " << map_load_path << std::endl;
                 } else {
                      std::cerr << "Warning: Label mapping file exists but is empty or invalid." << std::endl;
                 }
            }
        } else {
             std::cout << "Label map file not found: " << map_load_path << std::endl;
        }
        
        // Load is successful if the model loaded OR the map loaded (allows detection if model is present)
        // Or perhaps better: require both for full functionality? Let's require model for now.
        return is_trained_; 

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error loading model/map: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error loading model/map: " << e.what() << std::endl;
    }
    return false; // Return false on exception
}

// Predicts the name for a given face ROI.
std::string FaceRecognizer::predict(const cv::Mat& face_roi) {
    if (!is_ready()) {
        // std::cerr << "Warning: Predict called but model is not ready." << std::endl;
        return "Unknown";
    }
    if (face_roi.empty()) {
         return "Unknown";
    }

    try {
        // Preprocess ROI
        cv::Mat resized_face_roi;
        cv::resize(face_roi, resized_face_roi, training_img_size_);
        cv::Mat flattened_face = resized_face_roi.reshape(1, 1);
        flattened_face.convertTo(flattened_face, CV_32F);

        // KNN Prediction
        cv::Mat results, neighborResponses, dists;
        float predicted_label_float = knn_model_->findNearest(flattened_face, knn_k_, results, neighborResponses, dists);
        int predicted_label = static_cast<int>(predicted_label_float);

        // Look up name
        if (label_to_name_map_.count(predicted_label) > 0) {
            return label_to_name_map_[predicted_label];
        }
    } catch (const cv::Exception& e) {
         std::cerr << "OpenCV Error during prediction: " << e.what() << std::endl;
    } catch (const std::exception& e) {
         std::cerr << "Standard Error during prediction: " << e.what() << std::endl;
    }
    
    return "Unknown"; // Return Unknown if label not found or error occurred
}

// Checks if the model is loaded, trained, and ready for prediction.
bool FaceRecognizer::is_ready() const {
    // Model must be loaded and trained, and map should ideally be non-empty
    return knn_model_ && is_trained_ && !label_to_name_map_.empty();
}

// Gets the current label-to-name map.
const std::unordered_map<int, std::string>& FaceRecognizer::get_label_map() const {
     return label_to_name_map_;
} 