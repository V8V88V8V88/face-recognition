import cv2
import os
import sys

CASCADE_PATH = 'models/haarcascade_frontalface_alt2.xml'

def main():
    headless = '--headless' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--headless']
    if not headless:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        print(f"Failed to load cascade from {CASCADE_PATH}")
        return

    source = args[0] if args else 0
    if source != 0:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        elif isinstance(source, str) and not os.path.exists(source):
            print(f"File not found: {source}")
            return
    vid = cv2.VideoCapture(source)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS) or 25
    writer = None
    if headless:
        writer = cv2.VideoWriter('output_faces.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if headless:
            writer.write(frame)
        else:
            cv2.imshow('Face Detection (OpenCV)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    if writer:
        writer.release()
        print('Written output_faces.mp4')
    else:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
