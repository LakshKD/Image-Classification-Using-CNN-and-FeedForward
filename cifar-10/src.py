from __future__ import print_function
import tensorflow as tf
import os
import glob
import nltk
import numpy as np
from scipy import misc
from PIL import Image
from nltk.tokenize import word_tokenize

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

file_train_1 = 'cifar-10-data/data_batch_1'
file_train_2 = 'cifar-10-data/data_batch_2'
file_train_3 = 'cifar-10-data/data_batch_3'
file_train_4 = 'cifar-10-data/data_batch_4'
file_train_5 = 'cifar-10-data/data_batch_5'
file_test  = 'cifar-10-data/test_batch'

d1 = unpickle(file_train_1)    
d2 = unpickle(file_train_2)    
d3 = unpickle(file_train_3)    
d4 = unpickle(file_train_4)    
d5 = unpickle(file_train_5)
d6 = unpickle(file_test)

Imagelist_train = []
Imagelist_test  = []

#img = d1["data"][0]
#temp = img.reshape(3,1024)
#R = temp[0].reshape(32,32)+128
#G = temp[1].reshape(32,32)+128
#B = temp[2].reshape(32,32)+128
#fimage = np.dstack((R,G,B))
#Imagelist_train.append(fimage)
#data=np.asarray(Imagelist_train)[0]
#print (data)
#img = Image.fromarray(data, 'RGB')
#img.save('my.png')
#print(d1["labels"][0])

################################################Training Images############################################
for img in d1["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_train.append(fimage)
for img in d2["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_train.append(fimage)
for img in d3["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_train.append(fimage)
for img in d4["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_train.append(fimage)
for img in d5["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_train.append(fimage)

#################################################Test Images##############################################
for img in d6["data"]:
    temp = img.reshape(3,1024)
    R = temp[0].reshape(32,32)
    G = temp[1].reshape(32,32)
    B = temp[2].reshape(32,32)
    fimage = np.dstack((R,G,B))
    Imagelist_test.append(fimage)    

###################################################Label#################################################
train_label =  []
test_label = []

for labels in d1["labels"]:
    train_label.append(labels)
for labels in d2["labels"]:
    train_label.append(labels)
for labels in d3["labels"]:
    train_label.append(labels)
for labels in d4["labels"]:
    train_label.append(labels)
for labels in d5["labels"]:
    train_label.append(labels)

for labels in d6["labels"]:
    test_label.append(labels)


train_label = np.asarray(train_label)
trainonehot = np.zeros([50000,10])
j=0
for i in train_label:
    trainonehot[j][i] = 1
    j = j+1


test_label = np.asarray(test_label)
testonehot = np.zeros([10000,10])
j=0
for i in test_label:
    testonehot[j][i] = 1
    j = j+1


#data=np.asarray(Imagelist_train)[0]
#img = Image.fromarray(data, 'RGB')
#img.save('my.png')
#print(d1["labels"][0])

images,labels_vec,images_valid,labels_valid_vec = np.asarray(Imagelist_train),trainonehot,np.asarray(Imagelist_test),testonehot

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 32,32,3])
y_ = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32)
 
#x_image = tf.reshape(x, [-1, 32, 32, 3])
#x_image = tf.image.per_image_standardization(x)

W_conv1 = weight_variable([5, 5, 3, 32],'W_conv1')
b_conv1 = bias_variable([32],'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64],'W_conv2')
b_conv2 = bias_variable([64],'b_conv2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*64])


W_fc1 = weight_variable([2*2*64, 1024],'W_fc1') 
b_fc1 = bias_variable([1024],'b_fc1')
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024,256],'W_fc2') 
b_fc2 = bias_variable([256],'b_fc2')
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)


W_fc3 = weight_variable([256, 10],'W_fc3')
b_fc3 = bias_variable([10],'b_fc3')
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, tf.to_int64(y_, name='ToInt64')))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.to_int64(y_, name='ToInt64'))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

chk1=tf.argmax(y_conv, 1)
chk2=tf.to_int64(y_, name='ToInt64')




sess = tf.InteractiveSession()
saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())
for i in range(1000000):
    batch_xs = images[(i%500)*100:((i%500)*100)+100]
    batch_ys = train_label[(i%500)*100:((i%500)*100)+100]
    feed_dict = {x:batch_xs, y_:batch_ys, keep_prob:0.4} # dropout = 1-keep
    sess.run(train_step, feed_dict = feed_dict)
    if((i%5000)==0):
        print("accuracy   :  ",)
        print(accuracy.eval({x:images, y_:train_label, keep_prob:1.0}))
        print("error   :  ",)
        print(cross_entropy.eval({x:images, y_:train_label, keep_prob:1.0}))
        print("chk1   :  ",)
        print(chk1.eval({x:images[0:10], y_:train_label[0:10], keep_prob:1.0}))
        print("chk2   :  ",)
        print(chk2.eval({x:images[0:10], y_:train_label[0:10], keep_prob:1.0}))
        save_path = saver.save(sess, '/home/development/lakshya/mtp_matter/DataMiningProject/Imagetagging/cifar-10/model.ckpt')



print(accuracy.eval({x:images, y_:labels_vec, keep_prob:1.0}))
print(accuracy.eval({x:images_valid, y_:labels_valid_vec, keep_prob:1.0}))
save_path = saver.save(sess, '/home/development/lakshya/mtp_matter/DataMiningProject/Imagetagging/cifar-10/model.ckpt')
print("Model saved in file : %s" % save_path)

