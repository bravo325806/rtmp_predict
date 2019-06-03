import numpy as np
import cv2
import multiprocessing as mp
import time
from darkflow.net.build import TFNet

def show_rectangle(frame, colors, labels, results):
    for result in results:
        label = '{}: {:.0f}%'.format(result['label'], result['confidence'] * 100)
        cv2.rectangle(frame, (result['topleft']['x'],result['topleft']['y']), (result['bottomright']['x'],result['bottomright']['y']), colors[labels.index(result['label'])], 4)
        cv2.putText(frame, label, (result['topleft']['x'],result['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)
    cv2.imshow('original', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

def camThread(frameBuffer, results, labels, colors):
    cap = cv2.VideoCapture()
    cap.open('rtsp://10.0.0.174:8554/channel=0/subtype=0/vod=20180921-123456')
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1366, 768))
        if frameBuffer.full():
            frameBuffer.get()

        frameBuffer.put(frame.copy())

        if not results.empty():
            result = results.get()
            show_rectangle(result['img'], colors, labels, result['box'])

def inferencer(frameBuffer, results):
    options = {"pbLoad": "built_graph/tiny-yolo-voc.pb", "metaLoad": "built_graph/tiny-yolo-voc.meta", "threshold": 0.4}
    tfnet = TFNet(options)
    while True:
        if not frameBuffer.empty():
            image = frameBuffer.get()
            outputs = tfnet.return_predict(image)
            boxInfo ={'img':image, 'box':outputs}
            results.put(boxInfo)
         

def load_labels():
    labels = []
    with open('labels.txt') as f:
         for i in f.readlines():
             labels.append(i.replace("\n", ""))
    return labels


if __name__ == '__main__':
   process = []
   colors =[tuple(255 * np.random.rand(3)) for i in range(20)]
   labels = load_labels()
 
   try:
       mp.set_start_method('forkserver', force=True)
       frameBuffer = mp.Queue(8)
       results = mp.Queue()

       p = mp.Process(target=inferencer, args=(frameBuffer, results,), daemon=False)
       p.start()
       process.append(p)

       p = mp.Process(target=camThread, args=(frameBuffer, results, labels, colors,), daemon=False)
       p.start()
       process.append(p)

       while True:
           time.sleep(1)

   except:
      import traceback
      traceback.print_exc()
   finally:
      for p in process:
          p.terminate()
