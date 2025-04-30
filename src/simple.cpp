#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>

int main() {
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape_predictor;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> shape_predictor;

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error opening the webcam." << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error capturing frame." << std::endl;
            break;
        }

        dlib::cv_image<dlib::bgr_pixel> dlib_frame(frame);

        std::vector<dlib::rectangle> faces = detector(dlib_frame);

        for (const auto& face : faces) {
            cv::rectangle(frame, cv::Point(face.left(), face.top()), cv::Point(face.right(), face.bottom()), cv::Scalar(255, 0, 0), 2);
        }

        cv::imshow("Live Face Detection", frame);

        if (cv::waitKey(1) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

