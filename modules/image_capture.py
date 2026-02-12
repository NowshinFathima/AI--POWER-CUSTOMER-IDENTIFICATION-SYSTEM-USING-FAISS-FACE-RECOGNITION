import cv2

def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()
    return frame
