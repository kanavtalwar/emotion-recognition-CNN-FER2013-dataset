import cv2
import os
import numpy as np

class Face:
    def __init__(self):
        self.haar_xml = os.path.join(os.getcwd(), 'face_detect', 'haar_cascade\haar_cascade_frontalface.xml')
        self.cascade_classifier = cv2.CascadeClassifier(self.haar_xml)
        self.cam = cv2.VideoCapture(0)

    def detectFace(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)

        return faces

    def WebCamFaceDetect(self, model):
        while True:
            ret, img = self.cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detectFace(img)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
                roi_gray = gray[y:y+h, x:x+w]
                cv2.putText(img, model.predict(roi_gray), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Live Feed", img)
            if(cv2.waitKey(1) == ord('q')):
                break

        self.cam.release()
        cv2.destroyAllWindows()