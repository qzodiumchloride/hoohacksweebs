# code borrowed from the book Neural Network Projects with Python by James Loy


import cv2
import os

face_cascade = cv2.CascadeClassifier(
    'prediction model/haarcascade_frontalface_default.xml')


def detect_faces(img, draw_box=True):
    # convert image to grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    face_box, face_coords = None, []

    for (x, y, w, h) in faces:
        if draw_box:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        face_box = img[y:y+h, x:x+w]
        face_coords = [x, y, w, h]

    return img, face_box, face_coords


# if __name__ == "__main__":
#     files = os.listdir('prediction model/sample_faces')
#     images = [file for file in files if 'jpg' in file]
#     for image in images:
#         img = cv2.imread('prediction model/sample_faces/' + image)
#         detected_face, _, _ = detect_faces(img)
#         cv2.imwrite('prediction model/sample_faces/detected_faces/' +
#                     image, detected_face)
