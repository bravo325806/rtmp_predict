import cv2
from darkflow.net.build import TFNet

def show_rectangle(frame, colors, labels, outputs):
    for output in outputs:
        label = '{}: {:.0f}%'.format(output['label'], output['confidence'] * 100)
        cv2.rectangle(frame, (output['topleft']['x'],output['topleft']['y']), (output['bottomright']['x'],output['bottomright']['y']), colors[labels.index(output['label'])], 4)
        cv2.putText(frame, label, (output['topleft']['x'],output['topleft']['y']), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 4)
    return frame
#    cv2.imshow('original', frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        pass

def inferencer(frameBuffer, colors, labels, results):
    options = {"pbLoad": "built_graph/tiny-yolo-voc.pb", "metaLoad": "built_graph/tiny-yolo-voc.meta", "threshold": 0.35}
    tfnet = TFNet(options)
    while True:        
       if not frameBuffer.empty():
           image = frameBuffer.get()
           outputs = tfnet.return_predict(image)
      # boxInfo ={'img':image, 'box':outputs}
           result = show_rectangle(image, colors, labels, outputs)
           if results.full():
               results.get()
           results.put(result)

