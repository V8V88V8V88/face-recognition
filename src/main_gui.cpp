#include <gtkmm.h>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <atomic>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace fs = std::filesystem;

const std::string FACES_DATA_FOLDER = "./data/faces/";
const std::string MODELS_FOLDER = "./models/";
const std::string FACE_DETECTION_MODEL_FILE = MODELS_FOLDER + "face_detection_yunet_2023mar.onnx";
const std::string TRAINED_MODEL_FILE = MODELS_FOLDER + "trained_model.yml";
const std::string LABEL_MAPPING_FILE = MODELS_FOLDER + "label_mapping.txt";

const int REQUIRED_CAPTURES = 10;
const double RECOGNITION_THRESHOLD = 65.0;

class FaceRecognitionWindow : public Gtk::Window {
private:
    // GUI members
    Gtk::Box m_main_box;
    Gtk::Box m_button_box;
    Gtk::Image m_video_display;
    Gtk::Entry m_name_entry;
    Gtk::Button m_capture_button;
    Gtk::Button m_train_button;
    Gtk::Button m_detect_button;
    Gtk::Label m_info_label;

    // OpenCV members
    cv::VideoCapture m_video_capture;
    cv::Ptr<cv::FaceDetectorYN> m_face_detector;
    cv::Ptr<cv::face::LBPHFaceRecognizer> m_face_recognizer;
    std::unordered_map<int, std::string> m_label_to_name_map;

    // State variables
    int m_captures_taken = 0;
    std::atomic<bool> m_is_detecting{false};
    sigc::connection m_timer_connection;

public:
    FaceRecognitionWindow() 
        : m_main_box(Gtk::Orientation::VERTICAL, 10),
          m_button_box(Gtk::Orientation::HORIZONTAL, 5),
          m_capture_button("Capture (0/" + std::to_string(REQUIRED_CAPTURES) + ")"),
          m_train_button("Train Model"),
          m_detect_button("Start Detection"),
          m_info_label("Enter name and capture images, or train existing data.")
    {
        set_title("Face Recognition Training & Detection");
        set_default_size(800, 600);

        // Initialize face detection (YuNet) and recognition
        m_face_detector = cv::FaceDetectorYN::create(
            FACE_DETECTION_MODEL_FILE,
            "", // No config file needed for YuNet ONNX
            cv::Size(320, 320), // Input size expected by model (can be adjusted)
            0.9f, // Score threshold
            0.3f, // NMS threshold
            5000  // Top K detections
        );

        if (!m_face_detector) { // Check if pointer is valid
             std::cerr << "Error loading YuNet face detector model!" << std::endl;
             update_info_label("Error: Could not load face detection model!");
             return; // Or handle error appropriately
        }
        // Set input size for the detector based on frame size later if needed, or keep fixed
        // m_face_detector->setInputSize(cv::Size(frame.cols, frame.rows)); // Example if dynamic size needed

        m_face_recognizer = cv::face::LBPHFaceRecognizer::create();
        
        // Try to load existing model
        load_recognizer();

        // Setup GUI
        set_child(m_main_box);
        m_main_box.set_margin(10);
        m_main_box.set_margin_start(10);
        m_main_box.set_margin_end(10);
        m_main_box.set_margin_top(10);
        m_main_box.set_margin_bottom(10);

        // Video display setup
        m_video_display.set_vexpand(true);
        m_video_display.set_hexpand(true);
        m_video_display.set_size_request(640, 480);
        m_main_box.append(m_video_display);

        // Info label
        m_info_label.set_margin_top(5);
        m_main_box.append(m_info_label);

        // Button box setup
        m_button_box.set_halign(Gtk::Align::CENTER);
        m_main_box.append(m_button_box);

        m_name_entry.set_placeholder_text("Enter Name");
        m_button_box.append(m_name_entry);

        // Connect signals
        m_name_entry.signal_changed().connect([this]() {
            m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
        });

        m_capture_button.set_sensitive(false);
        m_capture_button.signal_clicked().connect(
            sigc::mem_fun(*this, &FaceRecognitionWindow::on_capture_button_clicked));
        m_button_box.append(m_capture_button);

        m_train_button.signal_clicked().connect(
            sigc::mem_fun(*this, &FaceRecognitionWindow::on_train_button_clicked));
        m_button_box.append(m_train_button);

        m_detect_button.signal_clicked().connect(
            sigc::mem_fun(*this, &FaceRecognitionWindow::on_detect_button_clicked));
        m_button_box.append(m_detect_button);

        // Initialize camera
        if (!init_camera()) {
            update_info_label("Error: Could not initialize camera!");
            return;
        }

        // Start video feed timer
        m_timer_connection = Glib::signal_timeout().connect(
            sigc::mem_fun(*this, &FaceRecognitionWindow::on_timeout), 33);
    }

    bool init_camera() {
        int max_retries = 3;
        for (int i = 0; i < max_retries; i++) {
            std::cout << "Attempting to open camera (attempt " << (i + 1) << " of " << max_retries << ")..." << std::endl;
            
            m_video_capture.open(0);
            if (m_video_capture.isOpened()) {
                m_video_capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                m_video_capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
                m_video_capture.set(cv::CAP_PROP_FPS, 30);
                
                cv::Mat test_frame;
                if (m_video_capture.read(test_frame) && !test_frame.empty()) {
                    std::cout << "Camera initialized successfully." << std::endl;
                    return true;
                }
                m_video_capture.release();
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return false;
    }

    void save_captured_face(const cv::Mat& frame, const cv::Rect& face_rect, const std::string& name) {
        cv::Mat face = frame(face_rect).clone();
        cv::Mat gray_face;
        cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        cv::resize(gray_face, gray_face, cv::Size(100, 100));

        std::string filename = FACES_DATA_FOLDER + name + "_" + 
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".jpg";
        
        if (cv::imwrite(filename, gray_face)) {
            std::cout << "Saved face image to: " << filename << std::endl;
        } else {
            std::cerr << "Error saving image to: " << filename << std::endl;
            update_info_label("Error saving captured image!");
        }
    }

    void on_capture_button_clicked() {
        std::string name = m_name_entry.get_text();
        if (name.empty()) {
            update_info_label("Please enter a name first.");
            return;
        }

        cv::Mat frame;
        if (!m_video_capture.read(frame)) {
            update_info_label("Error capturing frame!");
            return;
        }

        // --- Use YuNet for detection ---
        cv::Mat faces_mat; // Output matrix for detections
        // Ensure detector input size matches frame or is set
        m_face_detector->setInputSize(frame.size()); 
        m_face_detector->detect(frame, faces_mat);

        if (!faces_mat.empty() && faces_mat.rows > 0) {
            // Find the face with the highest score (confidence)
            int best_face_idx = 0;
            float highest_score = 0.0f;
            for (int i = 0; i < faces_mat.rows; ++i) {
                 float score = faces_mat.at<float>(i, 14); // Confidence score is at index 14
                 if (score > highest_score) {
                     highest_score = score;
                     best_face_idx = i;
                 }
            }

            // Extract bounding box for the best face
            // Box coordinates are floats at indices 0-3
            int x = static_cast<int>(faces_mat.at<float>(best_face_idx, 0));
            int y = static_cast<int>(faces_mat.at<float>(best_face_idx, 1));
            int w = static_cast<int>(faces_mat.at<float>(best_face_idx, 2));
            int h = static_cast<int>(faces_mat.at<float>(best_face_idx, 3));
            cv::Rect best_face_rect(x, y, w, h);

            // Ensure rect is within frame boundaries
            best_face_rect &= cv::Rect(0, 0, frame.cols, frame.rows); 
            if (best_face_rect.width <= 0 || best_face_rect.height <= 0) {
                 update_info_label("Detected face rect invalid. Try again.");
                 return;
            }


            save_captured_face(frame, best_face_rect, name);
            m_captures_taken++;
            
            m_capture_button.set_label("Capture (" + 
                std::to_string(m_captures_taken) + "/" + 
                std::to_string(REQUIRED_CAPTURES) + ")");

            if (m_captures_taken >= REQUIRED_CAPTURES) {
                update_info_label("Captured all required images for " + name + ". Ready to train!");
                m_capture_button.set_sensitive(false);
                m_captures_taken = 0; // Reset for next person
            }
        } else {
            update_info_label("No face detected! Please ensure your face is clearly visible.");
        }
    }

    void on_train_button_clicked() {
        update_info_label("Training started...");
        
        std::vector<cv::Mat> faces;
        std::vector<int> labels;
        std::unordered_map<std::string, int> name_to_label;
        m_label_to_name_map.clear();
        int next_label = 0;

        try {
            for (const auto& entry : fs::directory_iterator(FACES_DATA_FOLDER)) {
                if (entry.path().extension() != ".jpg") continue;

                std::string filename = entry.path().stem().string();
                size_t pos = filename.find_last_of('_');
                if (pos == std::string::npos) continue;

                std::string name = filename.substr(0, pos);
                cv::Mat face = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
                
                if (face.empty()) {
                    std::cerr << "Could not read image: " << entry.path() << std::endl;
                    continue;
                }

                int label;
                if (name_to_label.find(name) == name_to_label.end()) {
                    label = next_label++;
                    name_to_label[name] = label;
                    m_label_to_name_map[label] = name;
                    std::cout << "Assigned label " << label << " to " << name << std::endl;
                } else {
                    label = name_to_label[name];
                }

                faces.push_back(face);
                labels.push_back(label);
            }

            if (faces.empty()) {
                update_info_label("No training images found!");
                return;
            }

            m_face_recognizer->train(faces, labels);
            m_face_recognizer->save(TRAINED_MODEL_FILE);

            // Save label mapping
            std::ofstream mapping_file(LABEL_MAPPING_FILE);
            for (const auto& pair : m_label_to_name_map) {
                mapping_file << pair.first << "," << pair.second << "\n";
            }

            update_info_label("Training complete with " + std::to_string(faces.size()) + 
                            " images for " + std::to_string(next_label) + " people.");
            
        } catch (const std::exception& e) {
            std::cerr << "Error during training: " << e.what() << std::endl;
            update_info_label("Error during training!");
        }
    }

    void load_recognizer() {
        try {
            if (fs::exists(TRAINED_MODEL_FILE)) {
                m_face_recognizer->read(TRAINED_MODEL_FILE);
                
                if (fs::exists(LABEL_MAPPING_FILE)) {
                    std::ifstream mapping_file(LABEL_MAPPING_FILE);
                    std::string line;
                    while (std::getline(mapping_file, line)) {
                        std::istringstream iss(line);
                        std::string label_str, name;
                        if (std::getline(iss, label_str, ',') && std::getline(iss, name)) {
                            m_label_to_name_map[std::stoi(label_str)] = name;
                        }
                    }
                    update_info_label("Loaded model with " + 
                        std::to_string(m_label_to_name_map.size()) + " people.");
                }
            }
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
        }
    }

    void on_detect_button_clicked() {
        if (!m_is_detecting) {
            if (m_label_to_name_map.empty()) {
                update_info_label("Please train the model first!");
                return;
            }
            m_is_detecting = true;
            m_detect_button.set_label("Stop Detection");
            m_capture_button.set_sensitive(false);
            m_train_button.set_sensitive(false);
            m_name_entry.set_sensitive(false);
        } else {
            m_is_detecting = false;
            m_detect_button.set_label("Start Detection");
            m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
            m_train_button.set_sensitive(true);
            m_name_entry.set_sensitive(true);
        }
    }

    bool on_timeout() {
        cv::Mat frame;
        if (!m_video_capture.read(frame)) {
            std::cerr << "Error reading frame" << std::endl;
            return true;
        }

        cv::Mat display_frame = frame.clone();

        if (m_is_detecting) {
            // --- Use YuNet for detection ---
            cv::Mat faces_mat; // Output matrix for detections
            // Ensure detector input size matches frame or is set
            m_face_detector->setInputSize(frame.size()); 
            m_face_detector->detect(frame, faces_mat);

            if (!faces_mat.empty() && faces_mat.rows > 0) {
                 for (int i = 0; i < faces_mat.rows; ++i) {
                     // Extract bounding box
                     int x = static_cast<int>(faces_mat.at<float>(i, 0));
                     int y = static_cast<int>(faces_mat.at<float>(i, 1));
                     int w = static_cast<int>(faces_mat.at<float>(i, 2));
                     int h = static_cast<int>(faces_mat.at<float>(i, 3));
                     cv::Rect face_rect(x, y, w, h);

                     // Ensure rect is within frame boundaries
                     face_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
                     if (face_rect.width <= 0 || face_rect.height <= 0) continue; // Skip invalid rect

                     // --- Perform Recognition (using the detected ROI) ---
                     cv::Mat gray;
                     cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // Convert whole frame once
                     cv::Mat face_roi = gray(face_rect); // Extract ROI from gray frame

                     // Resize ROI for the recognizer
                     cv::Mat resized_face_roi;
                     cv::resize(face_roi, resized_face_roi, cv::Size(100, 100));

                     int label = -1;
                     double confidence = 0.0;
                     
                     try {
                         m_face_recognizer->predict(resized_face_roi, label, confidence);
                         
                         std::string name = "Unknown";
                         cv::Scalar color(0, 0, 255); // Red for unknown

                         // Use recognition threshold and check label exists
                         if (confidence < RECOGNITION_THRESHOLD && m_label_to_name_map.count(label) > 0) {
                             name = m_label_to_name_map[label];
                             color = cv::Scalar(0, 255, 0); // Green for recognized
                         } else {
                              name = "Unknown"; // Explicitly set unknown if threshold not met or label missing
                              color = cv::Scalar(0, 0, 255); 
                         }


                         // --- Draw bounding box and label ---
                         cv::rectangle(display_frame, face_rect, color, 2);

                         std::ostringstream text;
                         text << name << " (" << std::fixed << std::setprecision(1) 
                              << (100.0 - confidence) << "%)"; // Use 100.0 for float division

                         int baseline = 0;
                         cv::Size text_size = cv::getTextSize(text.str(), 
                             cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline); // Slightly smaller font
                         
                         // Position text above the box
                         cv::Point text_org(face_rect.x, face_rect.y - 5); 
                         // Ensure text isn't drawn off-screen
                         if (text_org.y < text_size.height) {
                              text_org.y = face_rect.y + face_rect.height + text_size.height + 5; // Below if no space above
                         }
                          if (text_org.x < 0) text_org.x = 0;


                         // Simple black background rectangle for text
                         cv::rectangle(display_frame,
                             cv::Point(text_org.x, text_org.y - text_size.height - baseline),
                             cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                             cv::Scalar(0, 0, 0), -1);
                             
                         cv::putText(display_frame, text.str(), text_org,
                             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1); // White text
                         
                     } catch (const cv::Exception& e) {
                         std::cerr << "Error during face recognition: " << e.what() << std::endl;
                         // Draw basic box even if recognition fails
                         cv::rectangle(display_frame, face_rect, cv::Scalar(0, 255, 255), 2); // Yellow box on error
                     }
                } // End loop through faces
            } // End if faces detected
        } // End if m_is_detecting

        try {
            cv::Mat rgb_frame;
            cv::cvtColor(display_frame, rgb_frame, cv::COLOR_BGR2RGB);
            
            auto pixbuf = Gdk::Pixbuf::create_from_data(
                rgb_frame.data,
                Gdk::Colorspace::RGB,
                false,
                8,
                rgb_frame.cols,
                rgb_frame.rows,
                static_cast<int>(rgb_frame.step)
            );

            int widget_width = m_video_display.get_width();
            int widget_height = m_video_display.get_height();

            if (widget_width <= 0 || widget_height <= 0) {
                 widget_width = m_video_display.get_allocated_width() > 0 ? m_video_display.get_allocated_width() : rgb_frame.cols;
                 widget_height = m_video_display.get_allocated_height() > 0 ? m_video_display.get_allocated_height() : rgb_frame.rows;
            }

            double scale_x = static_cast<double>(widget_width) / pixbuf->get_width();
            double scale_y = static_cast<double>(widget_height) / pixbuf->get_height();
            double scale = std::min(scale_x, scale_y);
            scale = std::max(0.01, scale);

            int scaled_width = static_cast<int>(pixbuf->get_width() * scale);
            int scaled_height = static_cast<int>(pixbuf->get_height() * scale);

            if (scaled_width > 0 && scaled_height > 0) {
                auto scaled_pixbuf = pixbuf->scale_simple(
                    scaled_width,
                    scaled_height,
                    Gdk::InterpType::BILINEAR
                );

                if (scaled_pixbuf) {
                    m_video_display.set(scaled_pixbuf);
                }
            } else if (pixbuf) {
                m_video_display.set(pixbuf);
            }
        } catch (const Glib::Error& e) {
            std::cerr << "GDK/GTK Error updating display: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard Error updating display: " << e.what() << std::endl;
        }

        return true;
    }

    void update_info_label(const Glib::ustring& text) {
        m_info_label.set_text(text);
    }
};

int main(int argc, char* argv[]) {
    if (!fs::exists(FACES_DATA_FOLDER)) {
        fs::create_directories(FACES_DATA_FOLDER);
    }
    
    if (!fs::exists(MODELS_FOLDER)) {
        fs::create_directories(MODELS_FOLDER);
    }

    auto app = Gtk::Application::create("org.example.facerecognition");
    return app->make_window_and_run<FaceRecognitionWindow>(argc, argv);
} 