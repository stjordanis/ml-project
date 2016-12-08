from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np
import load_lfw

##############################################################################
#   DATA WRANGLING                                                           #
##############################################################################
#lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
lfw_people = load_lfw.load_pairs(type='funneled')
#print(lfw_people)
fold0 = lfw_people[0]
#print("fold0 = ", fold0)
# Find the shapes of the images.
#num_images, h, w = lfw_people.images.shape
#print(fold0)
# Extract the input to the neural network and the number of features.
#X = lfw_people.data
#print(X)
target_names = np.asarray([(fold0[x][0][0], fold0[x][1][0]) for x in range(len(fold0))])
#print(target_names)
print(target_names.shape)
#print([fold0[x][y][0] for x in range(len(fold0)) for y in range(0,2)])
target_nums = np.asarray([int(x==y) for (x,y) in target_names])
#target_nums = np.reshape(target_nums, (600,1))
print("target_nums.shape = ", target_nums.shape)
data = []
for x in range(len(fold0)):
	data.append((np.ravel(fold0[x][0][2]),np.ravel(fold0[x][1][2])))
X = np.asarray(data)
print("fold0[x][0][2].shape = " , fold0[x][0][2].shape)
print("fold0[x][1][2].shape = " , fold0[x][1][2].shape)
images =np.asarray([ np.concatenate((fold0[x][0][2], fold0[x][1][2]), axis=0) for x in range(len(fold0))])
images = np.reshape(images, (600, 500, 250, 1))
print("images.shape =", images.shape)
#X = np.asarray([fold0[x][y][2] for x in range(len(fold0)) for y in range(0,2)])
#X = np.ravel(X)
# (1200, 250, 250)
num_features = X.shape[2] #should be 62500
print("num_features = ", num_features)
print(X.shape)
print(images.shape) 
num_pairs = images.shape[0]
h = 250 # TODO hard-coded
w = 250
# Extract the target data.
#target_names = lfw_people.target_names
#lfw_people = load_lfw.load_pairs()
# num_classes = target_names.shape[0]
num_classes = 1 # since we are classifying whether the people are the same or different
print("num_classes = ", num_classes)

# TODO logic here to account for pairs 

t = np.zeros([num_pairs, num_classes])
#print(np.arange(num_pairs))
#t[np.arange(num_pairs), target_nums] = 1
t[np.arange(num_pairs), 0] = target_nums

print("Total dataset size:")
print("#image pairs: %d" % num_pairs)
print("#pixels: %d" % num_features)
print("#identities: %d" % num_classes)
# DOES X_train and X_test contain concatenated images??? (500, 250)
# Create training and test sets. 
X_train, X_test, t_train, t_test = train_test_split(images, t, test_size=0.25, random_state=0)
print("t_train.shape = ", t_train.shape)
print("X_train.shape = ", X_train.shape)
#print("t_train = ", t_train)
#print("t_test = ", t_test)
print("X_test.shape = ", X_test.shape)
print("t_test.shape = ", t_test.shape)
#print("X_test = ", X_test)
#print("t_test = ", t_test)
'''
t_train.shape =  (450, 2)
X_train.shape =  (450, 2, 62500)
X_test.shape =  (150, 2, 62500)
t_test.shape =  (150, 2)
'''

###############################################################################
#   NEURAL NETWORK CONSTANTS                                                  #
###############################################################################
# HYPERPARAMETERS
NUM_TRAINING_STEPS = 50

batch_size = 30
learning_rate = 1e-4
NUM_CHANNELS = 1
layer1_filter_size = 5
layer1_depth = 16
layer2_filter_size = 5
layer2_depth = 32

layer1_stride = 2
layer2_stride = 2

layer1_pool_filter_size = 2
layer1_pool_stride = 1

layer2_pool_filter_size = 2
layer2_pool_stride = 1


fully_connected_nodes = 30 #100


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def conv(x, W, sw=2, sh=2):
    return tf.nn.conv2d(x, W, strides=[1, sw, sh, 1], padding='SAME') # should this be strides=[sw, sh, 1, 1]

def max_pool(x, w=2, h=2, sw=2, sh=2):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, sw, sh, 1], padding='SAME')

###############################################################################
#   BUILD THE NEURAL NETWORK                                                  #
###############################################################################
# Input layer.
depth = 1 # not doing color right now
input_layer = tf.placeholder(tf.float32, [None, 2 * w, h, depth])
# input_layer = tf.placeholder(tf.float32, [None, w * h])
# input_layer_2d = tf.reshape(input_layer, [-1, w, h, 1])
# input_layer_2d = tf.reshape(input_layer, [-1, 2 * w, h])

print("input_layer.get_shape():")
print(input_layer.get_shape())

# Convolutional layer 1
conv1_w = weight_variable([layer1_filter_size, layer1_filter_size, 1, layer1_depth])
conv1_b = bias_variable([layer1_depth])
conv1_h = tf.nn.relu(conv(input_layer, conv1_w) + conv1_b)

print("conv1_h.get_shape()")
print(conv1_h.get_shape())

# Pooling layer 1
pool1_h = max_pool(conv1_h, w=2, h=2, sw=1, sh=1)

#print("h_pool1.get_shape()")
#print(h_pool1.get_shape())

# Convolutional layer.
conv2_w = weight_variable([layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth])
conv2_b = bias_variable([layer2_depth])
conv2_h = tf.nn.relu(conv(pool1_h, conv2_w) + conv2_b)

print("conv2_h.get_shape()")
print(conv2_h.get_shape())

# Pooling layer.
pool2_h = max_pool(conv2_h, w=2, h=2, sw=1, sh=1)
#h_pool2 = max_pool(h_conv2, w=layer2_pool_filter_size, h=layer2_pool_filter_size, sw=layer2_pool_stride, sh=layer2_pool_stride)
_, w, h, d = pool2_h.get_shape().as_list()
#_, pool2w, pool2h, pool2d = h_pool2.get_shape().as_list()
#units = pool2w * pool2h * pool2d
pool2_h_flat = tf.reshape(pool2_h, [-1, w*h*d])
#h_pool2_flat = tf.reshape(h_pool2, [-1, units])

print("pool2_h_flat.get_shape()")
print(pool2_h_flat.get_shape())

# Fully connected layer.
fc1_w = weight_variable([w*h*d, fully_connected_nodes])
fc1_b = bias_variable([fully_connected_nodes])
fc1_h = tf.nn.relu(tf.matmul(pool2_h_flat, fc1_w) + fc1_b)

print("fc1_h.get_shape()")
print(fc1_h.get_shape())

# Output layer.
out_w = weight_variable([fully_connected_nodes, num_classes])
out_b = bias_variable([num_classes])
# one of these
#output_layer = tf.nn.l2_normalize(tf.matmul(fc1_h, out_w) + out_b, 1)
output_layer = tf.sigmoid(tf.matmul(fc1_h, out_w) + out_b)
print_output = tf.Print(output_layer, [output_layer], message='Output values : ')

print("output_layer.get_shape()")
print(output_layer.get_shape())

# Layer storing the actual target values.
target_layer = tf.placeholder(tf.float32, [None, num_classes])
print_target = tf.Print(target_layer, [target_layer], message='Target values : ')
# TODO -- load this?
print("target_layer.get_shape()")
print(target_layer.get_shape())

# The loss function.
cross_entropy = tf.reduce_mean(tf.square(output_layer - target_layer)) # tf.nn.sigmoid_cross_entropy_with_logits(output_layer, target_layer) # tf.reduce_mean(
print_loss = tf.Print(cross_entropy, [cross_entropy], message='Loss values : ')

# tf.square(output_layer - target_layer) #
# The training step.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


###############################################################################
#   CALCULATE ACCURACY                                                        #
###############################################################################
def accuracy(X, t):
    #correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(target_layer,1))
    print("ACCURACY")
    print("X.shape = ", X.shape) 
    print("t.shape = ", t.shape)
   # ('X.shape = ', (150, 500, 250, 1))
   # ('t.shape = ', (150, 1))
    print("X = ", X)
    print("t = ", t)
    correct_prediction = tf.equal(tf.round(tf.to_float(output_layer), tf.to_float(target_layer)))
    print("correct_prediction = ", correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={target_layer:t, input_layer:X})
    return acc
    
###############################################################################
#   TRAIN THE NEURAL NETWORK                                                  #
###############################################################################
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
t0=time()
for i in range(NUM_TRAINING_STEPS):
    idxs = np.arange(0, t_train.shape[0])
    np.random.shuffle(idxs)

    X_batch = X_train[idxs[:batch_size]]
    t_batch = t_train[idxs[:batch_size]]
    #print("X_batch.shape = ", X_batch.shape)
    #print("X_train.shape = ", X_train.shape)
    #print("t_batch.shape = ", t_batch.shape)
    #print("t_train.shape = ", t_train.shape)


    _, loss_val, out, target, loss = sess.run([train_step, cross_entropy, print_output, print_target, print_loss], feed_dict={target_layer:t_batch, input_layer:X_batch})
    
    if i % 10 == 9:
        train_time = time() - t0
        print('Batch %d: %f seconds' % (i, train_time))
        t0=time()
       # print("loss_val = ", loss_val)
       # print("out = ", out)
       # print("target = ", target)
 

###############################################################################
#   EVALUATE                                                                  #
###############################################################################
#print('TEST ACCURACY: ')
#print(accuracy(X_test, t_test))

# TODO
# Single sigmoid output
# anchor + match + non-match
###############################################################################
#   RUN THE NEURAL NETWORK                                                    #
###############################################################################
print("Predicting people's names on the test set")
test = tf.round(output_layer)
y_pred = sess.run(test, feed_dict={input_layer:X_test})
print("y_pred = ", y_pred)
#t_pred = np.argmax(t_test, 1)
t_pred = t_test
print("t_pred = ", t_pred)

#print(classification_report(t_pred, y_pred, target_names=target_names))
print(classification_report(t_pred, y_pred))
#print(confusion_matrix(t_pred, y_pred, labels=range(num_classes)))
print(confusion_matrix(t_pred, y_pred))
