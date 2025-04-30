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
#include <mutex>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <glibmm/dispatcher.h>

namespace fs = std::filesystem;

const std::string FACES_DATA_FOLDER = "./data/faces/";

// Construct std::string objects for concatenation
const std::string MODELS_FOLDER_STR = MODELS_FOLDER; // Use the macro defined by CMake
const std::string FACE_CASCADE_FILE = MODELS_FOLDER_STR + "haarcascade_frontalface_alt2.xml";
const std::string MODEL_FILE = MODELS_FOLDER_STR + "face_model.yml";
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

    // --- State variables ---
    int m_captures_taken = 0;
    std::atomic<bool> m_is_detecting{false};

    // --- Threading Members ---
    std::thread m_processing_thread;
    std::atomic<bool> m_thread_running{false};
    std::atomic<bool> m_detection_enabled{false}; // Controls if detection/recognition runs
    std::mutex m_frame_mutex;
    cv::Mat m_latest_display_frame; // Frame ready for GUI display
    Glib::Dispatcher m_update_dispatcher; // Signals GUI thread to update display

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

        // Connect the dispatcher signal to the GUI update function
        m_update_dispatcher.connect(sigc::mem_fun(*this, &FaceRecognitionWindow::on_update_display_requested));

        // --- Try to load existing recognition data via FaceRecognizer ---
        if (m_recognizer.load(MODEL_FILE, LABEL_MAPPING_FILE)) {
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
        m_name_entry.signal_changed().connect([this]() { // Reset captures on name change
            m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
            m_captures_taken = 0;
            m_capture_button.set_label("Capture (0/" + std::to_string(REQUIRED_CAPTURES) + ")");
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

        // Initialize camera and start processing thread
        if (!init_camera()) {
            // Error already shown in init_camera
        }
    }

    // Destructor to clean up the thread
    ~FaceRecognitionWindow() override {
        m_thread_running = false; // Signal thread to stop
        if (m_processing_thread.joinable()) {
            m_processing_thread.join(); // Wait for thread to finish
        }
        if (m_video_capture.isOpened()) {
            m_video_capture.release(); // Release camera resource
        }
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
                    // Start the processing thread only if camera opened
                    m_thread_running = true;
                    m_processing_thread = std::thread(&FaceRecognitionWindow::process_frames, this);
                    return true;
                }
                m_video_capture.release();
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        update_info_label("Error: Could not initialize camera!");
        return false;
    }

    // --- Worker Thread Function ---
    void process_frames() {
        cv::Mat frame;
        cv::Mat display_frame; // Frame to potentially draw on
        cv::Mat gray;          // Grayscale frame for detection/recognition

        while (m_thread_running) {
            if (!m_video_capture.isOpened() || !m_video_capture.read(frame)) {
                std::cerr << "Error reading frame in worker thread" << std::endl;
                // Prevent busy-waiting on error
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (frame.empty()) {
                continue;
            }

            display_frame = frame.clone(); // Work on a copy for display annotations

            // --- Perform detection/recognition if enabled ---
            if (m_detection_enabled) {
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // Convert original frame
                std::vector<cv::Rect> faces;
                m_face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

                for (const auto& face_rect : faces) {
                    if (face_rect.width <= 0 || face_rect.height <= 0) continue;

                    cv::Mat face_roi = gray(face_rect);
                    if (face_roi.empty()) continue;

                    std::string name = "Unknown";
                    cv::Scalar color(0, 0, 255); // Default red

                    try {
                        // Use thread-safe predict if needed, assume ok for now
                        name = m_recognizer.predict(face_roi);
                        if (name != "Unknown") {
                            color = cv::Scalar(0, 255, 0); // Green if recognized
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "Error during prediction: " << e.what() << std::endl;
                        name = "Error";
                        color = cv::Scalar(0, 255, 255); // Yellow on error
                    }

                    // --- Draw bounding box and label on the display_frame ---
                    cv::rectangle(display_frame, face_rect, color, 2);
                    std::ostringstream text;
                    text << name;
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(text.str(),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                    cv::Point text_org(face_rect.x, face_rect.y - 10);
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
            } // End if m_detection_enabled

            // --- Update shared frame for GUI ---
            { // Mutex scope
                std::lock_guard<std::mutex> lock(m_frame_mutex);
                m_latest_display_frame = display_frame.clone(); // Store the annotated frame
            }

            // --- Signal GUI thread to update ---
            m_update_dispatcher.emit();

            // Yield or sleep briefly to prevent high CPU usage if needed
            std::this_thread::yield();
            // std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Optional small sleep
        }
        std::cout << "Processing thread finished." << std::endl;
    }

    void save_captured_face(const cv::Mat& frame, const cv::Rect& face_rect, const std::string& name) {
        // Check if face rectangle is reasonable size
        if (face_rect.width < 50 || face_rect.height < 50) {
            std::cerr << "Face too small for good quality training: " 
                      << face_rect.width << "x" << face_rect.height << std::endl;
            update_info_label("Face too small! Please move closer to the camera.");
            return;
        }
        
        // Extract face region
        cv::Mat face = frame(face_rect).clone();
        cv::Mat gray_face;
        cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        
        // Check for face quality - simple variance check
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray_face, mean, stddev);
        
        // Low variance might indicate poor lighting, blurry image, etc.
        if (stddev[0] < 30.0) {
            std::cerr << "Low variance in face image, likely poor quality: " << stddev[0] << std::endl;
            update_info_label("Poor lighting detected! Ensure face is well lit.");
            return;
        }
        
        // Apply preprocessing for consistency with recognition
        cv::equalizeHist(gray_face, gray_face);
        
        // Resize to standard size
        cv::Mat resized_face;
        cv::resize(gray_face, resized_face, cv::Size(100, 100));

        // Create a unique filename with timestamp
        std::string filename = FACES_DATA_FOLDER + name + "_" + 
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".jpg";
        
        if (cv::imwrite(filename, resized_face)) {
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

        // --- Clear data folder ONLY when starting captures for the current name ---
        if (m_captures_taken == 0) {
            std::cout << "Clearing data folder for new capture session: " << FACES_DATA_FOLDER << std::endl;
            try {
                if (fs::exists(FACES_DATA_FOLDER)) {
                    for (const auto& entry : fs::directory_iterator(FACES_DATA_FOLDER)) {
                        fs::remove(entry.path()); // Delete each file
                    }
                } else {
                     fs::create_directories(FACES_DATA_FOLDER); // Ensure it exists if deleted manually
                }
            } catch (const std::exception& e) {
                 std::cerr << "Error clearing data folder: " << e.what() << std::endl;
                 update_info_label("Error clearing old data!");
                 // Decide if we should proceed or stop?
                 return; 
            }
        }

        // Get the latest frame from the worker thread
        cv::Mat frame_for_capture;
        {
            std::lock_guard<std::mutex> lock(m_frame_mutex);
            if (m_latest_display_frame.empty()) {
                update_info_label("No frame available for capture yet.");
                return;
            }
            // Use the frame *before* potential annotations were drawn by detection
            // For simplicity, let's use the latest frame, which might have boxes.
            // A cleaner way would be to store both raw and display frame separately.
            frame_for_capture = m_latest_display_frame.clone();
        }
        if (frame_for_capture.empty()) { // Double check after clone
             return;
        }

        // --- Use Haar Cascade for detection ---
        cv::Mat gray;
        cv::cvtColor(frame_for_capture, gray, cv::COLOR_BGR2GRAY);
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

            save_captured_face(frame_for_capture, *largest_face, name);
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

        bool success = m_recognizer.train(FACES_DATA_FOLDER, MODEL_FILE, LABEL_MAPPING_FILE);

        if (success) {
            update_info_label("Training complete. Ready for detection!");
        } else {
            update_info_label("Training failed. Check console output.");
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
            
            m_detection_enabled = true; // Enable processing in worker thread
            
            m_is_detecting = true;
            m_detect_button.set_label("Stop Detection");
            m_capture_button.set_sensitive(false);
            m_train_button.set_sensitive(false);
            m_name_entry.set_sensitive(false);
        } else {
            m_detection_enabled = false; // Disable processing in worker thread
            m_is_detecting = false;
            m_detect_button.set_label("Start Detection");
            m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
            m_train_button.set_sensitive(true);
            m_name_entry.set_sensitive(true);
        }
    }

    // --- GUI Update Function (called by dispatcher on GUI thread) ---
    void on_update_display_requested() {
        cv::Mat frame_to_display;

        // Get the latest frame prepared by the worker thread
        {
            std::lock_guard<std::mutex> lock(m_frame_mutex);
            if (m_latest_display_frame.empty()) {
                return; // No frame ready yet
            }
            frame_to_display = m_latest_display_frame.clone();
        }

        if (frame_to_display.empty()) {
             return;
        }

        // Update the Gtk::Image widget
        try {
            cv::Mat rgb_frame;
            // Ensure conversion happens if needed (frame might be BGR)
            if (frame_to_display.channels() == 3) {
                 cv::cvtColor(frame_to_display, rgb_frame, cv::COLOR_BGR2RGB);
            } else if (frame_to_display.channels() == 1) {
                // Handle grayscale if necessary, though worker prepares BGR
                 cv::cvtColor(frame_to_display, rgb_frame, cv::COLOR_GRAY2RGB);
            } else {
                 rgb_frame = frame_to_display; // Assume it's already RGB? Or handle error.
            }

            auto pixbuf = Gdk::Pixbuf::create_from_data(
                rgb_frame.data,
                Gdk::Colorspace::RGB,
                false,
                8, // bits per sample
                rgb_frame.cols,
                rgb_frame.rows,
                static_cast<int>(rgb_frame.step)
            );

            // Scaling logic (keep aspect ratio)
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
    }

    void update_info_label(const Glib::ustring& text) {
        m_info_label.set_text(text);
    }
};

int main(int argc, char* argv[]) {
    // Ensure data and models directories exist (moved from training button click)
    if (!fs::exists(FACES_DATA_FOLDER)) {
        fs::create_directories(FACES_DATA_FOLDER);
    }
    if (!fs::exists(MODELS_FOLDER_STR)) { // Check MODELS_FOLDER_STR from CMake
        fs::create_directories(MODELS_FOLDER_STR);
    }

    auto app = Gtk::Application::create("org.example.facerecognition");
    return app->make_window_and_run<FaceRecognitionWindow>(argc, argv);
} 