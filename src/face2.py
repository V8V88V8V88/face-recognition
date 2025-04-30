import cv2
import face_recognition
import os
import threading
import queue

faces_folder = 'faces'
face_recognition_threshold = 0.6

known_face_encodings = []
known_face_names = []

def load_known_faces(faces_folder):
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(faces_folder):
        if filename.endswith(".jpg"):
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_folder, filename)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                encoding = face_encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)

def recognize_faces(frame):
    global known_face_encodings, known_face_names
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if face_distances[best_match_index] < face_recognition_threshold:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        face_names.append(name)

    return face_locations, face_names

def process_frame(frame_queue, result_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        face_locations, face_names = recognize_faces(frame)
        result_queue.put((frame, face_locations, face_names))

def main():
    load_known_faces(faces_folder)

    vid = cv2.VideoCapture(0)
    vid.set(3, 1280)
    vid.set(4, 720)

    frame_queue = queue.Queue()
    result_queue = queue.Queue()

    num_threads = 4
    threads = [threading.Thread(target=process_frame, args=(frame_queue, result_queue)) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        frame_queue.put(frame)
        result = result_queue.get()
        frame, face_locations, face_names = result

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for _ in range(num_threads):
        frame_queue.put(None)

    for thread in threads:
        thread.join()

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
