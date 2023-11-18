import cv2 

vid = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True: 
    ret, frame = vid.read() 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 

cv2.destroyAllWindows()
