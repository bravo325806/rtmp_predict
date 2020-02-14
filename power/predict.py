from openvino.inference_engine import IENetwork, IEPlugin
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
    image = image.transpose((2, 0, 1))
    image = image.reshape(1, 3, 30, 15)
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
    labels = load_labels()
    areas = load_area()
    cap = cv2.VideoCapture(0)
    center = (640/2,480/2)
    angle = 90
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    plugin = IEPlugin(device="MYRIAD")
    net = IENetwork(model="model/model.xml", weights="model/model.bin")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    exec_net = plugin.load(network=net)
    if not os.path.exists('logs'):
        os.mkdir('logs')
    crawl = 1440
    start = time.time()
    while crawl<1800:
        ret, frame = cap.read()
        result_number = ''
        count = 1
        frame = cv2.warpAffine(frame, M, (640,480))
        original_img = frame[:,80:560,:]
        rect_img = np.copy(original_img)
        for i in areas:
            image = preprocess(original_img, i)
            req_handle = exec_net.start_async(request_id=0, inputs={input_blob: image})
            status = req_handle.wait()
            result = req_handle.outputs[out_blob][0]
            if count == int(i['name'][0]):
                result_number = result_number+labels[np.where(result==np.max(result))[0][0]]
            else:
                result_number = result_number+'\n'+labels[np.where(result==np.max(result))[0][0]]
                count +=1
            cv2.rectangle(rect_img, (i['xmin'], i['ymin']), (i['xmax'], i['ymax']), (0,0,255), 1)
        print('\n'+result_number+'\n')
        hour_time = time.strftime("%H-%M-%S", time.localtime())
        #if crawl* 60 <1440*60+(time.time()-start):
        #    cv2.imwrite('img/'+str(crawl)+'.jpg', original_img)
        #    crawl += 1
        #    print(crawl)
        if hour_time == '20-30-00':
            export(result_number, original_img, rect_img)
