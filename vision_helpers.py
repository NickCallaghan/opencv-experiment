import cv2


def find_objects(harr_file, image_file):
    object_cascade = cv2.CascadeClassifier(harr_file)
    detected_objects = object_cascade.detectMultiScale(
        image_file, scaleFactor=1.10, minNeighbors=12)
    return detected_objects


def mark_objects(object_list, image_file, object_name):
    for x, y, w, h in object_list:
        # Update image putting rectangle round objects
        image_file = cv2.rectangle(image_file, (x, y), (x+w, y+h),
                                   (0, 255, 0), thickness=5)

        cv2.putText(image_file, object_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return image_file
