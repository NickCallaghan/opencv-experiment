import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    'haar/haarcascade_frontalface_default.xml')
palm_cascade = cv2.CascadeClassifier('haar/palm_v4.xml')
fist_cascade = cv2.CascadeClassifier('haar/fist_v3.xml')

while (True):
    # Capture frame by frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Object detection in frame
    detected_faces = face_cascade.detectMultiScale(grey, scaleFactor=1.10,
                                                   minNeighbors=11)

    detected_palms = palm_cascade.detectMultiScale(
        grey, scaleFactor=1.10, minNeighbors=11)

    detected_fists = fist_cascade.detectMultiScale(
        grey, scaleFactor=1.10, minNeighbors=11)

    print(detected_fists)

    for x, y, w, h in detected_faces:
        # Update original image putting rectangle round faces
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), thickness=5)

        cv2.putText(frame, 'Face Detected', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    for x, y, w, h in detected_palms:
        # Update original image putting rectangle round faces
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), thickness=5)

        cv2.putText(frame, 'Palm Detected', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    for x, y, w, h in detected_fists:
        # Update original image putting rectangle round faces
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), thickness=5)

        cv2.putText(frame, 'Fist Detected', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
