import numpy as np
import cv2 as cv
import time
import paho.mqtt.client as mqtt

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the TX2 onboard camera
cap = cv.VideoCapture(0)

#face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv.CascadeClassifier('/face_detect/haarcascades/haarcascade_frontalface_default.xml')
client = mqtt.Client()

def get_face(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    #frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]

        rc,png = cv.imencode('.png', face)
        face = png.tobytes()
        post_face(face)
        #filename = str(int(round(time.time() * 1000))) + '.png'
        #f = open('output/' + filename, 'w+b')
	#f.write(msg)
	#f.close()

def post_face(face):
    client.connect("local_mqtt_broker",1883,60)
    #client.connect("169.53.167.199",1883,60)
    client.publish("w251/hw03/faces/capture", payload=face, qos=1, retain=False);
    client.disconnect();

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # We don't use the color information, so might as well save space
    #gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    get_face(frame)
    # face detection and other logic goes here

