import numpy as np
import cv2 as cv

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
cap = cv.VideoCapture(0)

#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('/face_detect/haarcascades/haarcascade_frontalface_default.xml')

def get_face(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    frame_gray = cv.equalizeHist(frame_gray)

    #faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        face = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        #print(frame)
        # your logic goes here; for instance
        # cut out face from the frame..
        rc,png = cv.imencode('.png', face)
        msg = png.tobytes()
	#print(msg)
        f = open('/face_detect/output/foo.png', 'w+b')
	f.write(msg)
	f.close()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # We don't use the color information, so might as well save space
    #gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    get_face(frame)
    # face detection and other logic goes here

