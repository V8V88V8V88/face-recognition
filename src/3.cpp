#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <filesystem>
#include <iostream>

using namespace dlib;
using namespace std;

template <typename SUBNET> using my_fc_no_bias = fc_no_bias<128, SUBNET>;
template <typename SUBNET> using my_relu = relu<my_fc_no_bias<SUBNET>>;
template <typename SUBNET> using my_ifc = max_pool<3, 3, 2, 2, relu<fc_no_bias<32, SUBNET>>>;
template <typename SUBNET> using my_tanh = relu<fc<32, SUBNET>>;

using anet_type = loss_metric<fc_no_bias<128, my_tanh<my_ifc<my_relu<input_rgb_image_sized<150>>>>>>;
using face_recognition_model = anet_type;

int main() {
    try {
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor sp;
        deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
        face_recognition_model face_recognition_net;
        deserialize("dlib_face_recognition_resnet_model_v1.dat") >> face_recognition_net;

        std::vector<matrix<rgb_pixel>> face_chips;
        std::vector<string> face_names;

        for (const auto &entry : std::filesystem::directory_iterator("faces")) {
            matrix<rgb_pixel> img;
            load_image(img, entry.path().string());
            std::vector<rectangle> faces = detector(img);
            std::vector<full_object_detection> shapes;

            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = sp(img, faces[i]);
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                face_chips.push_back(move(face_chip));
                face_names.push_back(entry.path().stem().string());
            }
        }

        std::vector<matrix<float, 0, 1>> face_descriptors = face_recognition_net(face_chips);

        image_window win;
        for (size_t i = 0; i < face_descriptors.size(); ++i) {
            for (size_t j = i; j < face_descriptors.size(); ++j) {
                if (length(face_descriptors[i] - face_descriptors[j]) < 0.6) {
                    cout << "Found " << face_names[i] << " in the image!" << endl;
                    win.set_image(face_chips[i]);
                    cin.get();
                }
            }
        }
    }
    catch (std::exception& e) {
        cout << e.what() << endl;
    }
    return 0;
}

