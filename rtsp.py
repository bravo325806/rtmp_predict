import cv2

def camThread(frameBuffer):
    cap = cv2.VideoCapture()
    cap.open('rtsp://10.0.0.174:8554/channel=0/subtype=0/vod=20180921-123456')
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (1280, 720))
            if frameBuffer.full():
                frameBuffer.get()
            frameBuffer.put(frame.copy())

