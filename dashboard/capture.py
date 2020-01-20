from darkflow.net.build import TFNet
import cv2
import numpy as np
import tensorflow as tf
import time

def load_labels():
    labels = []
    with open('labels.txt') as f:
        for i in f.readlines():
            labels.append(i.replace("\n","")) 
    return labels

def show_rectangle(frame,results):
    for result in results:
        if result['confidence']>0.4:
            label = '{}: {:.0f}%'.format(result['label'], result['confidence'] * 100)
            cv2.rectangle(frame, (result['topleft']['x'],result['topleft']['y']), (result['bottomright']['x'],result['bottomright']['y']), colors[labels.index(result['label'])], 4)
            cv2.putText(frame, label, (result['topleft']['x'],result['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 4)
    cv2.imshow('original', frame)

colors = [tuple(255 * np.random.rand(3)) for i in range(20)]
options = {"pbLoad": "built_graph/tiny-yolo-voc.pb", "metaLoad": "built_graph/tiny-yolo-voc.meta", "threshold": 0.4}

tfnet = TFNet(options)
labels = load_labels()
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture()
cap.open('rtsp://10.0.0.174:8554/channel=0/subtype=0/vod=20180921-123456')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1366, 768))
    results = tfnet.return_predict(frame)
    show_rectangle(frame, results)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
