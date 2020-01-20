import tensorflow as tf
import numpy as np
import os
import cv2
from random import shuffle
from bs4 import BeautifulSoup

class model(object):
    base_lr = 0.0001
    max_lr = 0.001
    batch_size = 64
    train_ratio = 0.9
    input_node_name = 'input'
    output_node_name = 'output'
    num_classes = 10
    data_set = []
    train_set = []
    test_set = []
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 30, 15, 3], name=self.input_node_name)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes])
        self.get_xml()
        self.split_dataset()
        self.network()       
        self.train()
        self.summary()
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def get_xml(self):
        for root, dirs, files in os.walk('/home/cheng/project/object_detection/dataset/number/yolo/xml'):
            for file in files:
                if file == '.DS_Store':continue
                content = open(root+'/'+file)
                soup = BeautifulSoup(content, 'html.parser')

                identify = soup.find_all('filename')[0].get_text().split('.')[0]
                for i in range(len(soup.find_all('name'))):
                    bb_name = soup.find_all('name')[i].get_text()
                    xmin = soup.find_all('xmin')[i].get_text()
                    ymin = soup.find_all('ymin')[i].get_text()
                    xmax = soup.find_all('xmax')[i].get_text()
                    ymax = soup.find_all('ymax')[i].get_text()
                    self.data_set.append({'identify':identify, 'name':bb_name, 'xmin':int(xmin), 'ymin':int(ymin), 'xmax':int(xmax), 'ymax':int(ymax)}) 
                

    def get_list(self):
        for root, dirs, files in os.walk('../dataset/trainingSet'):
            for file in files:
                label = root.split('/')[-1]
                dic = {'label':label, 'file':root+"/"+file}
                self.data_set.append(dic)

    def split_dataset(self):
        train_size = int(len(self.data_set) * self.train_ratio)
        shuffle(self.data_set)
        self.train_set = self.data_set[0:train_size]
        self.test_set = self.data_set[train_size:]

    def resblock(self, input_x, filter):
        conv_1 = tf.layers.conv2d(inputs=input_x, filters=filter, kernel_size=[3,3], padding='same')
        conv_1 = tf.layers.batch_normalization(conv_1)
        conv_1 = tf.nn.relu(conv_1)
        conv_2 = tf.layers.conv2d(inputs=conv_1, filters=filter, kernel_size=[3,3], padding='same')
        conv_2 = tf.layers.batch_normalization(conv_2)
        return tf.nn.relu(tf.add(conv_1, conv_2))

    def network(self):
        for i in range(0,3):
            conv_1 = self.resblock(self.x, 32) if i==0 else self.resblock(conv_1, 32)
        for i in range(0,3):
            conv_2 = self.resblock(conv_1, 64) if i==0 else self.resblock(conv_2, 64)
        for i in range(0,3):
            conv_3 = self.resblock(conv_2, 128) if i==0 else self.resblock(conv_3, 128)
        pool_1 = tf.layers.average_pooling2d(inputs=conv_3, pool_size=[2,2], strides=2)     
        flatten = tf.layers.flatten(pool_1)
        fully = tf.layers.dense(flatten, 1024, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fully, 10)
        self.outputs = tf.nn.softmax(self.logits, name=self.output_node_name)
    
    def train(self):
        global_step = tf.Variable(0)
        step_size = int(len(self.train_set) / self.batch_size)
        cycle = tf.floor(1 + global_step/(2  * step_size))
        x = tf.cast(np.abs(global_step/step_size - 2 * cycle + 1), tf.float32)
        self.learning_rate = self.base_lr + (self.max_lr - self.base_lr) * tf.maximum(0., 1-x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=global_step)
        self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def summary(self): 
        summary_train_loss = tf.summary.scalar(name='train', tensor=self.loss, family="loss")
        summary_train_accuracy = tf.summary.scalar(name='train', tensor=self.accuracy, family="accuracy")
        summary_test_loss = tf.summary.scalar(name='test', tensor=self.loss, family="loss")
        summary_test_accuracy = tf.summary.scalar(name='test', tensor=self.accuracy, family="accuracy")
        self.merged_train_summary_op = tf.summary.merge([summary_train_loss, summary_train_accuracy])
        self.merged_test_summary_op = tf.summary.merge([summary_test_loss, summary_test_accuracy])

    def get_data(self, batch_size=64, is_training=True):
        batch_features = []
        labels = []
        epoch = 0
        path = '/home/cheng/project/object_detection/dataset/number/yolo/img/'
        data_set = self.train_set if is_training else self.test_set
        while True:
            shuffle(data_set)
            if is_training==True:
                epoch = epoch + 1
                print('epoch:', epoch)
            for data in data_set:
                image = cv2.imread(path+data['identify']+'.jpg', cv2.IMREAD_COLOR)
                image = image[data['ymin']:data['ymax'],data['xmin']:data['xmax'],:]
                resize_image = cv2.resize(image, (15,30), interpolation=cv2.INTER_CUBIC)
                rgb_img = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
                rgb_img = rgb_img/255.0
                batch_features.append(rgb_img)
                label = self.dense_to_one_hot(int(data['name']), self.num_classes)
                labels.append(label)
                if len(batch_features) >= batch_size:
                    yield np.array(batch_features), np.array(labels)
                    batch_features = []
                    labels = []

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        return np.eye(num_classes)[labels_dense]

    def save_graph_to_file(self, sess, path): 
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [self.output_node_name])
        with tf.gfile.FastGFile(path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())

MODEL_NAME = 'model'
net = model()
training_iters = int((50 * len(net.train_set)) / net.batch_size)
display_step = 1
with tf.Session() as sess:
    sess.run(net.init)
#    mnist_net.saver.restore(sess, "output/mnist")
    tf.train.write_graph(sess.graph_def, 'model', MODEL_NAME + '.pbtxt', True)
    summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
    step = 1
    train_batch = net.get_data(net.batch_size, True)
    test_batch = net.get_data(net.batch_size, False)
    while step < training_iters:
        batch_x, batch_y = next(train_batch)
        batch_a, batch_b = next(test_batch)
        sess.run(net.optimizer, feed_dict={net.x: batch_x, net.y: batch_y})
        if step % display_step == 0:
            train_acc = sess.run(net.accuracy, feed_dict={net.x: batch_x, net.y: batch_y})
            train_loss = sess.run(net.loss, feed_dict={net.x: batch_x, net.y: batch_y})
            test_acc = sess.run(net.accuracy, feed_dict={net.x: batch_a, net.y: batch_b})
            test_loss = sess.run(net.loss, feed_dict={net.x: batch_a, net.y: batch_b})
            print("Iter " + str(step) + ", Training Loss = " + \
                 "{:.6f}".format(train_loss) + ", Training Accuracy = " + \
                 "{:.5f}".format(train_acc) + ", Testbatch Loss = " + \
                 "{:.6f}".format(test_loss) + ", Testing Accuracy = " + \
                 "{:.5f}".format(test_acc))
        step = step + 1
        summary = sess.run(net.merged_train_summary_op, feed_dict={net.x: batch_x, net.y: batch_y})
        summary_writer.add_summary(summary, step)
        summary = sess.run(net.merged_test_summary_op, feed_dict={net.x: batch_a, net.y: batch_b})
        summary_writer.add_summary(summary, step)

    net.saver.save(sess, 'model/' + MODEL_NAME)
    net.save_graph_to_file(sess, 'model/'+MODEL_NAME+'.pb')
    print("Finish")
