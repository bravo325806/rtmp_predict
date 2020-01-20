import cv2
import time

cap = cv2.VideoCapture()
cap.open('rtmp://10.0.0.187:1935/demo/live')
center = (640/2,480/2)
angle = 90
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
count = 0
start = time.time()
while count<1440:
    ret, frame = cap.read()
    if ret:
        frame = cv2.warpAffine(frame, M, (640,480))
        frame = frame[:, 80:560,:]
        #cv2.imshow('frame',frame)
        if count*60 < (time.time()-start):
            cv2.imwrite('img/'+str(count)+'.jpg', frame)
            count+=1
            print(count)
