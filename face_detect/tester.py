import cv2
import numpy as np
import os
import face_detect as fr

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    faces_detected, gray_img = fr.face_detect(img)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv2.imshow("Face", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows