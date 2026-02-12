import cv2

def preprocess(face):
    face = cv2.resize(face, (160, 160))
    return face
