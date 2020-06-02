import face_recognition
import wget
import cv2
import numpy as np

shubham_image=wget.download(url="https://i.imgur.com/a1HmdVp.jpg",out='shubham.jpg')
shubham_image = face_recognition.load_image_file(shubham_image)
shubham_face_encoding = face_recognition.face_encodings(shubham_image)[0]

known_face_encodings = [
    shubham_face_encoding,
]
known_face_names = [
    "Shubham ", 
]

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]


    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    found=0
    for face_encoding in face_encodings:
        found=1
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.putText(frame, name, (left*4+6,top*4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
