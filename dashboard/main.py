import numpy as np
import cv2
import multiprocessing as mp
from flask import Flask, render_template, Response
from predict import inferencer
from rtsp import camThread

app = Flask(__name__)


def load_labels():
    labels = []
    with open('labels.txt') as f:
         for i in f.readlines():
             labels.append(i.replace("\n", ""))
    return labels

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    global results
    while True:
        if not results.empty():
#        cap = cv2.VideoCapture()
#        cap.open('rtsp://10.0.0.174:8554/channel=0/subtype=0/vod=20180921-123456')
#        ret, frame = cap.read()
            result = results.get()            
            ret, jpeg = cv2.imencode('.jpg', result)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   process = []
   colors =[tuple(255 * np.random.rand(3)) for i in range(20)]
   labels = load_labels()
 
   try:
       mp.set_start_method('forkserver', force=True)
       frameBuffer = mp.Queue(8)
       global results 
       results = mp.Queue(5)

       p = mp.Process(target=inferencer, args=(frameBuffer, colors, labels, results,), daemon=False)
       p.start()
       process.append(p)

       p = mp.Process(target=camThread, args=(frameBuffer,), daemon=False)
       p.start()
       process.append(p)
       app.run(host='0.0.0.0', debug=True)   

   except:
      import traceback
      traceback.print_exc()
   finally:
      for p in process:
          p.terminate()
