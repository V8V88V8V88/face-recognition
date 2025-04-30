import cv2
import face_recognition
import os
import threading
import queue

# Set QT_QPA_PLATFORM environment variable to use XCB
# This is often needed for OpenCV GUI functions on Wayland systems
os.environ['QT_QPA_PLATFORM'] = 'xcb'

faces_folder = 'data/faces'
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
    # Consider lowering resolution for better performance if needed
    # vid.set(3, 640)
    # vid.set(4, 480)
    vid.set(3, 1280)
    vid.set(4, 720)

    frame_queue = queue.Queue(maxsize=10) # Increase queue size slightly
    result_queue = queue.Queue()

    num_threads = 4 # Use a fixed number of threads
    threads = [threading.Thread(target=process_frame, args=(frame_queue, result_queue)) for _ in range(num_threads)]
    for thread in threads:
        thread.daemon = True # Make threads daemon so they exit when main exits
        thread.start()

    frame_count = 0
    FRAME_SKIP = 2 # Process every 2nd frame (adjust as needed)

    while True:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame, exiting.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            # Put frame into the queue for processing, but don't block if full
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                # Skip frame if the processing queue is full
                pass

        # Only display frames when a result is ready from the queue
        try:
            processed_frame, face_locations, face_names = result_queue.get(block=False)

            # Draw results on the *processed* frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since they were detected on scaled frame
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(processed_frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(processed_frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Display the resulting frame
            cv2.imshow('Face Recognition', processed_frame)

        except queue.Empty:
            # If no processed frame is ready, do nothing with display in this iteration
            pass

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Shutting down...")
    # Signal worker threads to exit
    for _ in range(num_threads):
        try:
            frame_queue.put(None, block=False) # Use non-blocking put
        except queue.Full:
            pass # Ignore if queue is full during shutdown signal

    # Wait for threads to finish (optional with daemon threads, but good practice)
    # for thread in threads:
    #     thread.join(timeout=1.0) # Add timeout to prevent hanging

    # Release handle to the webcam
    vid.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
