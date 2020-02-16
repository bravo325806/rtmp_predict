import tensorflow as tf
import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as dom
from bs4 import BeautifulSoup

def load_area(version):
    if version == 0:
        content = open('number/template/area-0.xml')
    elif version == 1440:
        content = open('number/template/area-1441.xml')
    elif version == 1950:
        content = open('number/template/area-1950.xml')
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

def load_labels():
    labels = []
    with open('output/labels.txt') as f:
         for i in f.readlines():
             labels.append(i.replace("\n", ""))
    return labels

def load_graph():
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open('output/model_4.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')

    return graph

def export(filename_index, block):
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    folder.text = "image"
    filename = ET.SubElement(annotation, 'filename')
    filename.text = str(filename_index) + ".jpg"
    path = ET.SubElement(annotation, 'path')
    path.text = path
    source = ET.SubElement(annotation, 'source')
    source.text = "Unknown"
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(480)
    height = ET.SubElement(size, 'height')
    height.text = str(480)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = "0"
    for b in block:
        box = ET.SubElement(annotation, 'object')
        name = ET.SubElement(box, 'name')
        name.text = b['name']
        pose = ET.SubElement(box, 'pose')
        pose.text = "Unspecified"
        truncated = ET.SubElement(box, 'truncated')
        truncated.text = "0"
        difficult = ET.SubElement(box, 'difficult')
        difficult.text = "0"
        bndbox = ET.SubElement(box, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(b['xmin'])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(b['ymin'])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(b['xmax'])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(b['ymax'])

    mydata = ET.tostring(annotation, 'utf-8')
    xml = dom.parseString(mydata)
    xml_pretty_str = xml.toprettyxml()
    myfile = open("number/xml/"+str(filename_index)+'.xml', "w")
    myfile.write(xml_pretty_str)

if __name__=='__main__':
    graph = load_graph()
    labels = load_labels()
    with tf.Session(graph=graph) as sess:
        x = sess.graph.get_tensor_by_name('input:0')
        output = sess.graph.get_tensor_by_name('output:0')
        files = os.listdir('number/img')
        files.sort()
        for file in files:
            path = 'number/xml/'+file.split('.')[0]+'.xml'
            if file == '.DS_Store' or os.path.isfile(path) :continue
            if int(file.split('.')[0])<1440: areas = load_area(0)
            elif int(file.split('.')[0])>1440 and int(file.split('.')[0])<1950: areas = load_area(1440)
            else: areas = load_area(1950)
            img = cv2.imread('number/img/'+file)
            result_number = ''
            box = list()
            for i in areas:
                image = img[i['ymin']:i['ymax'],i['xmin']:i['xmax']]
                image = cv2.resize(image, (15,30), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image/255.0
                result = sess.run(output, feed_dict={x:[image]})[0]
                if i['name'][0]=='5':
                    box.append({'name':labels[np.where(result==np.max(result))[0][0]], 'xmin': i['xmin'], 'xmax':i['xmax'], 'ymin':i['ymin'], 'ymax':i['ymax']})
                    result_number = result_number+labels[np.where(result==np.max(result))[0][0]]
            print(file)
            print(result_number+'\n')
            cv2.imshow('img', img)
            if cv2.waitKey(0) & 0xFF==ord('z'):
                export(file.split('.')[0], box)
