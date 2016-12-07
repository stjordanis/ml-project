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
# Find the shapes of the images.
#num_images, h, w = lfw_people.images.shape
#print(fold0)
# Extract the input to the neural network and the number of features.
#X = lfw_people.data
#print(X)
target_names = np.asarray([(fold0[x][0][0], fold0[x][1][0]) for x in range(len(fold0))])
print(target_names)
print(target_names.shape)
print([fold0[x][y][0] for x in range(len(fold0)) for y in range(0,2)])
target_nums = np.asarray([int(x==y) for (x,y) in target_names])
target_nums = np.reshape(target_nums, (600,1))
print(target_nums.shape)
data = []
for x in range(len(fold0)):
	data.append((np.ravel(fold0[x][0][2]),np.ravel(fold0[x][1][2])))
X = np.asarray(data)
images =np.asarray([(fold0[x][0][2], fold0[x][1][2]) for x in range(len(fold0))])
#X = np.asarray([fold0[x][y][2] for x in range(len(fold0)) for y in range(0,2)])
#X = np.ravel(X)
# (1200, 250, 250)
num_features = X.shape[1]
print("num_features = ", num_features)
print("num_features = ", num_features)
print(X.shape)
print(images.shape) 
num_pairs = images.shape[0]
h = images.shape[2]
w = images.shape[3]
# Extract the target data.
#target_names = lfw_people.target_names
#lfw_people = load_lfw.load_pairs()
# num_classes = target_names.shape[0]
num_classes = 2 # since we are classifying whether the people are the same or different
print("num_classes = ", num_classes)

# TODO logic here to account for pairs 

t = np.zeros([num_pairs, num_classes])
#print(np.arange(num_pairs))
t[np.arange(num_pairs), target_nums] = 1

print("Total dataset size:")
print("#image pairs: %d" % num_pairs)
print("#pixels: %d" % num_features)
print("#identities: %d" % num_classes)

# Create training and test sets.
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=0)
print("t_train.shape = ", t_train.shape)
print("X_train.shape = ", X_train.shape)
print("X_test.shape = ", X_test.shape)
print("t_test.shape = ", t_test.shape)

###############################################################################
#   NEURAL NETWORK CONSTANTS                                                  #
###############################################################################
# HYPERPARAMETERS
NUM_TRAINING_STEPS = 1000

batch_size = 30
learning_rate = 1e-4
NUM_CHANNELS = 1
layer1_filter_size = 5
layer1_depth = 32
layer2_filter_size = 5
layer2_depth = 64

layer1_stride = 2
layer2_stride = 2

layer1_pool_filter_size = 2
layer1_pool_stride = 1

layer2_pool_filter_size = 2
layer2_pool_stride = 1


fully_connected_nodes = 100


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def conv(x, W, sw, sh):
    return tf.nn.conv2d(x, W, strides=[sw, sh, 1, 1], padding='SAME') # should this be [1, sw, sh, 1]

def max_pool(x, w, h, sw, sh):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, sw, sh, 1], padding='SAME')

###############################################################################
#   BUILD THE NEURAL NETWORK                                                  #
###############################################################################
# Input layer.
input_layer = tf.placeholder(tf.float32, [None, w * h])
input_layer_2d = tf.reshape(input_layer, [-1, w, h, 1])

print("input_layer_2d.get_shape():")
print(input_layer_2d.get_shape())

# Convolutional layer 1
W_conv1 = weight_variable([layer1_filter_size, layer1_filter_size, 1, layer1_depth])
b_conv1 = bias_variable([layer1_depth])
h_conv1 = tf.nn.relu(conv(input_layer_2d, W_conv1, layer1_stride, layer1_stride) + b_conv1)

print("h_conv1.get_shape()")
print(h_conv1.get_shape())

# Pooling layer 1
h_conv1 = max_pool(h_conv1, w=layer1_pool_filter_size, h=layer1_pool_filter_size, sw=layer1_pool_stride, sh=layer1_pool_stride)

#print("h_pool1.get_shape()")
#print(h_pool1.get_shape())

# Convolutional layer.
W_conv2 = weight_variable([layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth])
b_conv2 = bias_variable([layer2_depth])
h_conv2 = tf.nn.relu(conv(h_conv1, W_conv2, layer2_stride, layer2_stride) + b_conv2)

print("h_conv2.get_shape()")
print(h_conv2.get_shape())

# Pooling layer.
h_pool2 = max_pool(h_conv2, w=layer2_pool_filter_size, h=layer2_pool_filter_size, sw=layer2_pool_stride, sh=layer2_pool_stride)
_, pool2w, pool2h, pool2d = h_pool2.get_shape().as_list()
units = pool2w * pool2h * pool2d
h_pool2_flat = tf.reshape(h_pool2, [-1, units])

print("h_pool2_flat.get_shape()")
print(h_pool2_flat.get_shape())

# Fully connected layer.
W_fc1 = weight_variable([units, fully_connected_nodes])
b_fc1 = bias_variable([fully_connected_nodes])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

print("h_fc1.get_shape()")
print(h_fc1.get_shape())

# Output layer.
W_output = weight_variable([fully_connected_nodes, num_classes])
b_output = bias_variable([num_classes])
output_layer = tf.sigmoid(tf.matmul(h_fc1, W_output) + b_output)

print("output_layer.get_shape()")
print(output_layer.get_shape())

# Layer storing the actual target values.
target_layer = tf.placeholder(tf.float32, [None, num_classes])
# TODO -- load this?


# The loss function.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, target_layer))

# The training step.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

###############################################################################
#   CALCULATE ACCURACY                                                        #
###############################################################################
def accuracy(X, t):
    correct_prediction = tf.equal(tf.argmax(output_layer,1), tf.argmax(target_layer,1))
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
    sess.run(train_step, feed_dict={target_layer:t_batch, input_layer:X_batch})
    
    if i % 10 == 9:
        train_time = time() - t0
        print('Batch %d: %f seconds' % (i, train_time))
        t0=time()


###############################################################################
#   EVALUATE                                                                  #
###############################################################################
print('TEST ACCURACY: ')
print(accuracy(X_test, t_test))

# TODO
# Single sigmoid output
# anchor + match + non-match
###############################################################################
#   RUN THE NEURAL NETWORK                                                    #
###############################################################################
print("Predicting people's names on the test set")
test = tf.argmax(output_layer, 1)
y_pred = sess.run(test, feed_dict={input_layer:X_test})
print(y_pred)
t_pred = np.argmax(t_test, 1)

print(classification_report(t_pred, y_pred, target_names=target_names))
print(confusion_matrix(t_pred, y_pred, labels=range(num_classes)))
