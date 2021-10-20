import cv2

haar_file = 'haarcascade_frontalface_default.xml'
print('load haar file..')
face_cascade = cv2.CascadeClassifier(haar_file)
print('haar file loaded..')


def detect_face(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 4)
    return [{'x': x, 'y': y, 'w': w, 'h': h} for (x,y,w,h) in faces]
    
