import cv2
import glob
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_dir = 'faces/'
face_files = os.listdir(face_dir)
total_images = len(face_files)
total_correct = 0


for file in face_files:
    img = cv2.imread(face_dir + file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = face_cascade.detectMultiScale(gray_img,
                                                   scaleFactor=1.10,
                                                   minNeighbors=11)

    for x, y, w, h in detected_faces:
        # Update original image putting rectangle round faces
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)

    # Resise Image
    scale_factor = 25
    width = int(img.shape[1] * scale_factor / 100)
    height = int(img.shape[0] * scale_factor / 100)
    dsize = (width, height)
    resized_image = cv2.resize(img, dsize)

    # Show Resized Image
    cv2.imshow("Image", resized_image)
    key_code = cv2.waitKeyEx()
    if key_code == 32:
        print("Image Correctly Identified")
        total_correct += 1
    else:
        print('Key Pressed: ', key_code)
    cv2.destroyAllWindows()

print(
    f'Scanning complete! Of {total_images} images, {total_correct} were correctly classied')
