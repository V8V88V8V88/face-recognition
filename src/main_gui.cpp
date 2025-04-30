#include <gtkmm.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include "FaceRecognizer.hpp"
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

// Construct std::string objects for concatenation
const std::string MODELS_FOLDER_STR = MODELS_FOLDER; // Use the macro defined by CMake
const std::string FACE_CASCADE_FILE = MODELS_FOLDER_STR + "haarcascade_frontalface_alt2.xml";
const std::string KNN_MODEL_FILE = MODELS_FOLDER_STR + "knn_model.yml";
const std::string LABEL_MAPPING_FILE = MODELS_FOLDER_STR + "label_mapping.txt";

const int REQUIRED_CAPTURES = 10;

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
    cv::CascadeClassifier m_face_cascade;
    FaceRecognizer m_recognizer;

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

        // Initialize face detection (Haar Cascade)
        if (!m_face_cascade.load(FACE_CASCADE_FILE)) {
             std::cerr << "Error loading cascade classifier" << std::endl;
             update_info_label("Error: Could not load face detection model!");
             return;
        }

        // --- Try to load existing recognition data via FaceRecognizer ---
        if (m_recognizer.load(KNN_MODEL_FILE, LABEL_MAPPING_FILE)) {
            update_info_label("Loaded existing recognition model.");
        } else {
             update_info_label("No model found. Train model after capturing.");
        }

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

        // --- Use Haar Cascade for detection ---
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        m_face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

        if (!faces.empty()) {
            // Find the largest face (original logic)
            auto largest_face = std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });
            
            // Ensure rect is valid (basic check)
            if (largest_face->width <= 0 || largest_face->height <= 0) {
                 update_info_label("Detected face rect invalid. Try again.");
                 return;
            }

            save_captured_face(frame, *largest_face, name);
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
        update_info_label("KNN Training started...");
        // Use a separate thread for training to avoid blocking GUI?
        // For simplicity, run directly for now.
        
        // Ensure data folder exists before calling train
        if (!fs::exists(FACES_DATA_FOLDER)) {
             fs::create_directories(FACES_DATA_FOLDER);
             update_info_label("Data folder created. Please capture images first.");
             return; // Can't train without data folder
        }
        if (!fs::is_directory(FACES_DATA_FOLDER)) {
             update_info_label("Error: Data path exists but is not a directory.");
             return;
        }

        // Ensure models folder exists
         if (!fs::exists(MODELS_FOLDER_STR)) {
             fs::create_directories(MODELS_FOLDER_STR);
         }

        bool success = m_recognizer.train(FACES_DATA_FOLDER, KNN_MODEL_FILE, LABEL_MAPPING_FILE);

        if (success) {
            update_info_label("KNN Training complete.");
        } else {
            update_info_label("KNN Training failed. Check console output.");
        }
    }

    void on_detect_button_clicked() {
        if (!m_is_detecting) {
            // Use the recognizer's readiness check
            if (!m_recognizer.is_ready()) {
                 update_info_label("Recognizer not ready. Please train model first!");
                 return;
            }
            // Check label map specifically (redundant if is_ready() includes it, but safe)
            if (m_recognizer.get_label_map().empty()) {
                update_info_label("Label map empty. Please train model first!");
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
        cv::Mat gray; // Declare gray frame once
        bool gray_converted = false;

        if (m_is_detecting) {
            // --- Use Haar Cascade for detection ---
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            gray_converted = true;
            std::vector<cv::Rect> faces;
            m_face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30)); 

            for (const auto& face_rect : faces) {
                if (face_rect.width <= 0 || face_rect.height <= 0) continue;
                
                cv::Mat face_roi = gray(face_rect); // Use the already converted gray frame
                if (face_roi.empty()) continue;
                
                // --- Perform Recognition using FaceRecognizer ---
                std::string name = "Unknown";
                cv::Scalar color(0, 0, 255); // Default red

                try {
                     name = m_recognizer.predict(face_roi);
                    
                     if (name != "Unknown") {
                          color = cv::Scalar(0, 255, 0); // Green if recognized
                     }
                } catch (const std::exception& e) {
                    std::cerr << "Error during prediction: " << e.what() << std::endl;
                    name = "Error";
                    color = cv::Scalar(0, 255, 255); // Yellow on error
                }

                // --- Draw bounding box and label ---
                cv::rectangle(display_frame, face_rect, color, 2);
                std::ostringstream text;
                text << name;
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(text.str(), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline); 
                cv::Point text_org(face_rect.x, face_rect.y - 10);
                // ... (Text positioning logic) ...
                 if (text_org.y < text_size.height) {
                         text_org.y = face_rect.y + face_rect.height + text_size.height + 5;
                    }
                 if (text_org.x < 0) text_org.x = 0;
                 if (text_org.x + text_size.width > display_frame.cols) {
                      text_org.x = display_frame.cols - text_size.width; 
                 }
                cv::rectangle(display_frame,
                    cv::Point(text_org.x, text_org.y - text_size.height - baseline - 5),
                    cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                    cv::Scalar(0, 0, 0), -1);
                cv::putText(display_frame, text.str(), text_org,
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            } // End loop through faces
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
    
    if (!fs::exists(MODELS_FOLDER_STR)) {
        fs::create_directories(MODELS_FOLDER_STR);
    }

    auto app = Gtk::Application::create("org.example.facerecognition");
    return app->make_window_and_run<FaceRecognitionWindow>(argc, argv);
} 