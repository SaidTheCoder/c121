from typing import final
import cv2
import time
import numpy as np
from numpy.core.numeric import outer

fourcc=cv2.VideoWriter_fourcc(*"XVID")
output=cv2.VideoWriter("output.avi",fourcc,20,(640,480))
cap=cv2.VideoCapture(0)
time.sleep(2)
background=0
for i in range(60):
    ret,background=cap.read()
background=np.flip(background,axis=1)
while(cap.isOpened()):
    ret,img=cap.read()
    if not ret:
        break
    img=np.flip(img,axis=1)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    red1=np.array([0,120,50])
    red2=np.array([10,255,255])
    mask1=cv2.inRange(hsv,red1,red2)
    red1=np.array([170,120,70])
    red2=np.array([180,255,255])
    mask2=cv2.inRange(hsv,red1,red2)
    mask1=mask1+mask2
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1=cv2.morphologyEx(mask1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    res1=cv2.bitwise_and(img,img,mask=mask2)
    res2=cv2.bitwise_and(background,background,mask=mask1)
    finaloutput=cv2.addWeighted(res1,1,res2,1,0)
    output.write(finaloutput)
    cv2.imshow("magic",finaloutput)
    cv2.waitKey(1)
cap.release()
output.release()
cv2.destroyAllWindows()