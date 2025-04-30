import cv2
import face_recognition
import os

def load_known_faces(faces_folder):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(faces_folder):
        if filename.endswith(".jpg"):
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(faces_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(person_name)

    return known_face_encodings, known_face_names

def recognize_faces(known_face_encodings, known_face_names):
    vid = cv2.VideoCapture('test.mp4')
    vid.set(3, 1280) 
    vid.set(4, 720)  

    while True:
        ret, frame = vid.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            name = known_face_names[best_match_index]
            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

faces_folder = 'faces'
known_face_encodings, known_face_names = load_known_faces(faces_folder)
recognize_faces(known_face_encodings, known_face_names)
