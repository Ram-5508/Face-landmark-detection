# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:05:21 2019

@author: Lenovo
"""

import cv2
import dlib

cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('C:/Users/Lenovo/Desktop/Ram_Perceptrons/Dlib/shape_predictor_68_face_landmarks.dat')

while 1:
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        landmark=predictor(gray,face)
        for i in range(0,68):
            x=landmark.part(i).x
            y=landmark.part(i).y
            cv2.circle(img,(x,y),3,(0,255,0),-1)
        
    cv2.imshow('Detect',img)
    k=cv2.waitKey(1)& 0xff
    if k==27:
        break
    
cap.release()
cv2.destroyAllWindows()