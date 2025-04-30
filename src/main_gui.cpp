#include <gtkmm.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <thread> // For potential background processing
#include <atomic> // For thread safety

namespace fs = std::filesystem;

const std::string FACES_DATA_FOLDER = "./data/faces/";
const std::string MODELS_FOLDER = "./models/";
const std::string MODEL_CONFIG = MODELS_FOLDER + "deploy.prototxt.txt";
const std::string MODEL_BINARY = MODELS_FOLDER + "res10_300x300_ssd_iter_140000.caffemodel";
const int REQUIRED_CAPTURES = 10;
const double LBPH_THRESHOLD = 80.0; // Adjusted threshold
const double PREDICTION_CONFIDENCE_THRESHOLD = 65.0; // Adjusted confidence check

class FaceRecognitionWindow : public Gtk::Window {
public:
    FaceRecognitionWindow();
    virtual ~FaceRecognitionWindow();

protected:
    // Signal handlers
    void on_capture_button_clicked();
    void on_train_button_clicked();
    void on_detect_button_clicked();
    bool on_timer_timeout(); // For updating the video feed

    // Member widgets
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
    cv::dnn::Net m_face_detector_net;
    cv::Ptr<cv::face::LBPHFaceRecognizer> m_face_recognizer;
    std::unordered_map<int, std::string> m_label_to_name_map; // Maps internal LBPH label index to person name

    // State variables
    int m_captures_taken = 0;
    std::atomic<bool> m_is_detecting{false};
    sigc::connection m_timer_connection;

    // Helper methods
    void update_frame();
    void load_recognizer();
    void save_captured_face(const cv::Mat& frame, const cv::Rect& face_rect, const std::string& name);
    void update_info_label(const std::string& message);
};

FaceRecognitionWindow::FaceRecognitionWindow()
    : m_main_box(Gtk::Orientation::VERTICAL, 10),
      m_button_box(Gtk::Orientation::HORIZONTAL, 5),
      m_name_entry(),
      m_capture_button("Capture (0/" + std::to_string(REQUIRED_CAPTURES) + ")"),
      m_train_button("Train Model"),
      m_detect_button("Start Detection"),
      m_info_label("Enter name and capture images, or train existing data.")
{
    set_title("Face Recognition Training & Detection");
    set_default_size(800, 600);
    set_child(m_main_box);

    m_main_box.set_margin(10);

    // Video Display Area (using Gtk::Image)
    m_video_display.set_vexpand(true);
    m_video_display.set_hexpand(true);
    m_main_box.append(m_video_display);

    // Info Label
    m_info_label.set_margin_top(5);
    m_main_box.append(m_info_label);

    // Button Box Setup
    m_button_box.set_halign(Gtk::Align::CENTER);
    m_main_box.append(m_button_box);

    m_name_entry.set_placeholder_text("Enter Name");
    m_button_box.append(m_name_entry);

    m_capture_button.set_sensitive(false); // Disabled until name is entered
    m_name_entry.signal_changed().connect([this]() {
        m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
    });
    m_capture_button.signal_clicked().connect(sigc::mem_fun(*this, &FaceRecognitionWindow::on_capture_button_clicked));
    m_button_box.append(m_capture_button);

    m_train_button.signal_clicked().connect(sigc::mem_fun(*this, &FaceRecognitionWindow::on_train_button_clicked));
    m_button_box.append(m_train_button);

    m_detect_button.signal_clicked().connect(sigc::mem_fun(*this, &FaceRecognitionWindow::on_detect_button_clicked));
    m_button_box.append(m_detect_button);

    // Initialize OpenCV components
    try {
        m_face_detector_net = cv::dnn::readNetFromCaffe(MODEL_CONFIG, MODEL_BINARY);
    } catch (const cv::Exception& ex) {
        std::cerr << "Error loading face detector model: " << ex.what() << std::endl;
        update_info_label("Error: Failed to load face detection model files from models/ directory!");
        // Disable buttons if model loading fails?
    }

    m_face_recognizer = cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8, LBPH_THRESHOLD);

    // Ensure data directory exists
    if (!fs::exists(FACES_DATA_FOLDER)) {
        fs::create_directories(FACES_DATA_FOLDER);
    }

    // Try to open camera
    m_video_capture.open(0);
    if (!m_video_capture.isOpened()) {
        std::cerr << "Error: Couldn't open the camera." << std::endl;
        update_info_label("Error: Cannot open webcam!");
        // Disable relevant buttons?
    } else {
        // Start timer for video feed updates (e.g., 30 FPS -> ~33ms)
        m_timer_connection = Glib::signal_timeout().connect(
            sigc::mem_fun(*this, &FaceRecognitionWindow::on_timer_timeout), 33);
    }

    load_recognizer(); // Try to load existing trained data
}

FaceRecognitionWindow::~FaceRecognitionWindow() {}

void FaceRecognitionWindow::update_info_label(const std::string& message) {
    m_info_label.set_text(message);
}

// --- Placeholder Signal Handlers --- 

void FaceRecognitionWindow::on_capture_button_clicked() {
    std::string name = m_name_entry.get_text();
    if (name.empty()) {
        update_info_label("Please enter a name first.");
        return;
    }
    if (m_captures_taken >= REQUIRED_CAPTURES) {
        update_info_label("Captured required images for " + name + ". Train or enter a new name.");
        return;
    }

    cv::Mat frame;
    if (!m_video_capture.read(frame) || frame.empty()) {
        update_info_label("Error capturing frame.");
        return;
    }

    // --- Face Detection Logic (Similar to previous code) ---
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
    m_face_detector_net.setInput(blob);
    cv::Mat detections = m_face_detector_net.forward();
    float max_confidence = 0;
    cv::Rect best_face_rect;

    for (int i = 0; i < detections.size[2]; ++i) {
        float confidence = detections.ptr<float>(0)[i * 7 + 2];
        if (confidence > 0.6) { // Use a slightly higher confidence for capture
             if(confidence > max_confidence) {
                max_confidence = confidence;
                int x1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);
                 // Clamp coordinates
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                x2 = std::min(frame.cols -1, x2);
                y2 = std::min(frame.rows -1, y2);
                 if (x2 > x1 && y2 > y1) {
                     best_face_rect = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                 }
             }
        }
    }

    if (max_confidence > 0.6 && !best_face_rect.empty()) {
        save_captured_face(frame, best_face_rect, name);
        m_captures_taken++;
        m_capture_button.set_label("Capture (" + std::to_string(m_captures_taken) + "/" + std::to_string(REQUIRED_CAPTURES) + ")");
        update_info_label("Captured image " + std::to_string(m_captures_taken) + " for " + name);
        if(m_captures_taken >= REQUIRED_CAPTURES) {
            update_info_label("Finished capturing for " + name + ". Ready to train.");
            m_capture_button.set_sensitive(false);
        }
    } else {
        update_info_label("No face detected clearly for capture.");
    }
}

void FaceRecognitionWindow::save_captured_face(const cv::Mat& frame, const cv::Rect& face_rect, const std::string& name) {
    cv::Mat face_roi = frame(face_rect);
    cv::Mat gray_face;
    cv::cvtColor(face_roi, gray_face, cv::COLOR_BGR2GRAY);
    cv::resize(gray_face, gray_face, cv::Size(200, 200)); // Larger size for saving?

    // Create unique filename
    long long timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::string filename = FACES_DATA_FOLDER + name + "_" + std::to_string(timestamp) + ".jpg";

    if (cv::imwrite(filename, gray_face)) {
        std::cout << "Saved captured face to: " << filename << std::endl;
    } else {
        std::cerr << "Error saving image: " << filename << std::endl;
        update_info_label("Error saving captured image.");
    }
}


void FaceRecognitionWindow::on_train_button_clicked() {
    update_info_label("Training started...");
    std::vector<cv::Mat> knownFaceImages;
    std::vector<int> knownFaceLabels;
    std::unordered_map<std::string, int> name_to_label_map;
    m_label_to_name_map.clear(); // Clear previous mapping
    int current_label = 0;

    for (const auto& entry : fs::directory_iterator(FACES_DATA_FOLDER)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
            std::string full_path = entry.path().string();
            std::string filename = entry.path().stem().string();

            // Extract name (assuming format NAME_timestamp)
            size_t separator_pos = filename.find_last_of('_');
            if (separator_pos == std::string::npos) {
                std::cerr << "Warning: Skipping file with unexpected format: " << filename << std::endl;
                continue;
            }
            std::string personName = filename.substr(0, separator_pos);

            cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Error: Couldn't read image " << full_path << std::endl;
                continue;
            }

            cv::resize(image, image, cv::Size(100, 100)); // Ensure consistent size for training

            int label;
            if (name_to_label_map.find(personName) == name_to_label_map.end()) {
                label = current_label++;
                name_to_label_map[personName] = label;
                m_label_to_name_map[label] = personName; // Store reverse mapping
                 std::cout << "Assigning label " << label << " to " << personName << std::endl;
            } else {
                label = name_to_label_map[personName];
            }

            knownFaceImages.push_back(image);
            knownFaceLabels.push_back(label);
        }
    }

    if (knownFaceImages.empty()) {
        update_info_label("Error: No face images found in " + FACES_DATA_FOLDER + " to train.");
        return;
    }

    m_face_recognizer->train(knownFaceImages, knownFaceLabels);
    update_info_label("Training complete with " + std::to_string(knownFaceImages.size()) + " images for " + std::to_string(current_label) + " people.");
    std::cout << "Training complete." << std::endl;
    m_captures_taken = 0; // Reset capture count after training
    m_capture_button.set_label("Capture (0/" + std::to_string(REQUIRED_CAPTURES) + ")");

    // Optional: Save the trained model
    // m_face_recognizer->save("lbph_trained_model.yml");
}

void FaceRecognitionWindow::load_recognizer() {
    // Optional: Load a previously saved model
    // try {
    //     m_face_recognizer->read("lbph_trained_model.yml");
    //     std::cout << "Loaded previously trained LBPH model." << std::endl;
    //     update_info_label("Loaded previously trained model.");
         // TODO: Need to reconstruct m_label_to_name_map if loading saved model
    // } catch (const cv::Exception& ex) {
    //     std::cerr << "No saved model found or error loading: " << ex.what() << std::endl;
    // }
}


void FaceRecognitionWindow::on_detect_button_clicked() {
    if (!m_is_detecting) {
        m_is_detecting = true;
        m_detect_button.set_label("Stop Detection");
        m_capture_button.set_sensitive(false);
        m_train_button.set_sensitive(false);
        m_name_entry.set_sensitive(false);
        update_info_label("Detection active.");
    } else {
        m_is_detecting = false;
        m_detect_button.set_label("Start Detection");
        m_capture_button.set_sensitive(!m_name_entry.get_text().empty());
        m_train_button.set_sensitive(true);
        m_name_entry.set_sensitive(true);
        update_info_label("Detection stopped.");
    }
}

// --- Video Feed Update --- 

bool FaceRecognitionWindow::on_timer_timeout() {
    if (!m_video_capture.isOpened()) {
        return false; // Stop timer if camera closed
    }

    cv::Mat frame;
    if (!m_video_capture.read(frame) || frame.empty()) {
        std::cerr << "Error reading frame from camera." << std::endl;
        return true; // Keep timer running
    }

    update_frame(); // Call the main update logic
    return true; // Keep timer running
}

void FaceRecognitionWindow::update_frame() {
    cv::Mat frame;
    if (!m_video_capture.read(frame) || frame.empty()) return;

    cv::Mat display_frame = frame.clone();

    // --- Detection Logic (Only if m_is_detecting is true) ---
    if (m_is_detecting) {
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123), false, false);
        m_face_detector_net.setInput(blob);
        cv::Mat detections = m_face_detector_net.forward();

        for (int i = 0; i < detections.size[2]; ++i) {
            float confidence = detections.ptr<float>(0)[i * 7 + 2];
            if (confidence > 0.5) {
                int x1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 3] * frame.cols);
                int y1 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 4] * frame.rows);
                int x2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 5] * frame.cols);
                int y2 = static_cast<int>(detections.ptr<float>(0)[i * 7 + 6] * frame.rows);

                // Clamp coordinates
                x1 = std::max(0, x1);
                y1 = std::max(0, y1);
                x2 = std::min(frame.cols -1, x2);
                y2 = std::min(frame.rows -1, y2);

                if (x2 > x1 && y2 > y1) {
                    cv::Rect faceRect(x1, y1, x2 - x1, y2 - y1);
                    cv::Mat faceROI = frame(faceRect);
                    cv::Mat grayROI;
                    cv::cvtColor(faceROI, grayROI, cv::COLOR_BGR2GRAY);
                    cv::resize(grayROI, grayROI, cv::Size(100, 100));

                    int predictedLabel = -1;
                    double predictionConfidence = 0.0;
                    m_face_recognizer->predict(grayROI, predictedLabel, predictionConfidence);

                    std::string personName = "Unknown";
                    if (predictedLabel != -1 && predictionConfidence < PREDICTION_CONFIDENCE_THRESHOLD) {
                         if(m_label_to_name_map.count(predictedLabel)) {
                            personName = m_label_to_name_map[predictedLabel];
                         }
                    }

                    // Draw rectangle and text on the display_frame
                    cv::rectangle(display_frame, faceRect, cv::Scalar(0, 255, 0), 2);
                    std::string label_text = personName + " (" + std::to_string(predictionConfidence) + ")";
                    cv::putText(display_frame, label_text, cv::Point(faceRect.x, faceRect.y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
            }
        }
    }

    // Convert OpenCV Mat to GdkPixbuf for display
    cv::cvtColor(display_frame, display_frame, cv::COLOR_BGR2RGB); // GTK expects RGB
    auto pixbuf = Gdk::Pixbuf::create_from_data(display_frame.data, Gdk::Colorspace::RGB,
                                                 false, 8, display_frame.cols, display_frame.rows, display_frame.step);
    
    // Scale pixbuf to fit the image widget while maintaining aspect ratio
    int widget_width = m_video_display.get_width();
    int widget_height = m_video_display.get_height();
    
    if (widget_width > 0 && widget_height > 0) {
        double scale_x = (double)widget_width / pixbuf->get_width();
        double scale_y = (double)widget_height / pixbuf->get_height();
        double scale = std::min(scale_x, scale_y); // Maintain aspect ratio

        int scaled_width = static_cast<int>(pixbuf->get_width() * scale);
        int scaled_height = static_cast<int>(pixbuf->get_height() * scale);

        // Ensure scaled dimensions are valid
        if (scaled_width > 0 && scaled_height > 0) {
            auto scaled_pixbuf = pixbuf->scale_simple(scaled_width, scaled_height, Gdk::InterpType::BILINEAR);
            m_video_display.set(scaled_pixbuf);
        } else {
             m_video_display.set(pixbuf); // Fallback if scaling results in zero size
        }

    } else {
         m_video_display.set(pixbuf); // Set unscaled if widget size is unknown
    }
}

// --- Main Function --- 

int main(int argc, char* argv[]) {
    auto app = Gtk::Application::create("org.example.facerecognition");

    return app->make_window_and_run<FaceRecognitionWindow>(argc, argv);
} 