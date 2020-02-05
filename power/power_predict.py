import tensorflow as tf
import os
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
from bs4 import BeautifulSoup

def load_labels():
    labels = []
    with open('model/labels.txt') as f:
         for i in f.readlines():
             labels.append(i.replace("\n", ""))
    return labels

def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open('model/model.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph

def load_area():
    content = open('model/area.xml')
    soup = BeautifulSoup(content, 'html.parser')

    identify = soup.find_all('filename')[0].get_text().split('.')[0]
    img_path = soup.find_all('path')[0].get_text()
    width = soup.find_all('width')[0].get_text()
    height = soup.find_all('height')[0].get_text()
    tmp = list()
    for i in range(len(soup.find_all('name'))):
        bb_name = soup.find_all('name')[i].get_text()
        xmin = soup.find_all('xmin')[i].get_text()
        ymin = soup.find_all('ymin')[i].get_text()
        xmax = soup.find_all('xmax')[i].get_text()
        ymax = soup.find_all('ymax')[i].get_text()
        tmp.append({'name':bb_name, 'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)})
    return tmp

def preprocess(image, area):
    image = image[area['ymin']:area['ymax'],area['xmin']:area['xmax']]
    image = cv2.resize(image, (15,30), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    return np.copy(image)

def export(result, original_img, rect_img):
    date = time.strftime("%Y-%m-%d", time.localtime())
    power = result.split('\n')[-1]
    if not os.path.exists('logs/'+date):
        os.mkdir('logs/'+date)
    cv2.imwrite('logs/'+date+'/original_img.jpg', original_img)
    cv2.imwrite('logs/'+date+'/rect_img.jpg', rect_img)
    f = open('logs/'+date+'/power.txt','w')
    f.write(power)
    f.close()
    MQTT = "10.20.0.19"
    MQTT_Port = 1883 #port
    MQTT_Topic = "Camera/Power" #TOPIC name
    k = "{\"camera_today\":"+power+"}"
    mqttc = mqtt.Client("python_pub")
    try:
        mqttc.connect(MQTT, MQTT_Port, 10)
        mqttc.publish(MQTT_Topic, k)
        print(date+' connect ok')
    except:
        print('mqtt connect error')
 
if __name__ =='__main__':
    graph = load_graph()
    labels = load_labels()
    areas = load_area()
    cap = cv2.VideoCapture()
    cap.open('rtmp://10.0.0.166:1935/demo/live')
    center = (640/2,480/2)
    angle = 90
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    show_rect = False
    if not os.path.exists('logs'):
        os.mkdir('logs')
    with tf.Session(graph=graph) as sess:
        x = sess.graph.get_tensor_by_name('input:0')
        output = sess.graph.get_tensor_by_name('output:0')
        while True:
            ret, frame = cap.read()
            if ret:
                result_number = ''
                count = 1
                frame = cv2.warpAffine(frame, M, (640,480))
                original_img = frame[:,80:560,:]
                rect_img = np.copy(original_img)
                for i in areas:
                    image = preprocess(original_img, i)
                    result = sess.run(output, feed_dict={x:[image]})[0]
                    if count == int(i['name'][0]):
                        result_number = result_number+labels[np.where(result==np.max(result))[0][0]]
                    else:
                        result_number = result_number+'\n'+labels[np.where(result==np.max(result))[0][0]]
                        count +=1
                    cv2.rectangle(rect_img, (i['xmin'], i['ymin']), (i['xmax'], i['ymax']), (0,0,255), 1)
                #print('\n'+result_number+'\n')
                hour_time = time.strftime("%H-%M-%S", time.localtime())
                if hour_time == '14-50-00':
                    export(result_number, original_img, rect_img)
