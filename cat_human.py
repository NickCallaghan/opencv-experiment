import numpy as np
import cv2
from vision_helpers import find_objects, mark_objects

#

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    faces_haar = 'haar/haarcascade_frontalface_default.xml'
    cat_haar = 'haar/haarcascade_frontalcatface_extended.xml'

    while (True):
        # Capture frame by frame
        ret, frame = cap.read()

        # Convert frame to greyscale to imporve face detection
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Look for human faces in frame and mark
        faces_list = find_objects(faces_haar, grey)
        frame = mark_objects(faces_list, frame, 'Human')

        # Looks for cat faces in the frame and mark
        cat_list = find_objects(cat_haar, grey)
        frame = mark_objects(cat_list, frame, 'Cat')

        # Display the resulting frame
        cv2.imshow('Detecting Faces', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
