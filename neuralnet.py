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
fold0 = lfw_people[0]

# Extract the input to the neural network and the number of features.

target_names = np.asarray([(fold0[x][0][0], fold0[x][1][0]) for x in range(len(fold0))])

target_nums = np.asarray([int(x==y) for (x,y) in target_names])

data = []
for x in range(len(fold0)):
	data.append((np.ravel(fold0[x][0][2]),np.ravel(fold0[x][1][2])))
X = np.asarray(data)

images =np.asarray([ np.concatenate((fold0[x][0][2], fold0[x][1][2]), axis=0) for x in range(len(fold0))])
images = np.reshape(images, (600, 500, 250, 1))

num_features = X.shape[2] #should be 62500
num_pairs = images.shape[0]
h = 250 # TODO hard-coded
w = 250
# Extract the target data.
#target_names = lfw_people.target_names
#lfw_people = load_lfw.load_pairs()
# num_classes = target_names.shape[0]
num_classes = 1 # since we are classifying whether the people are the same or different
print("num_classes = ", num_classes)


t = np.zeros([num_pairs, num_classes])
t[np.arange(num_pairs), 0] = target_nums

print("Total dataset size:")
print("#image pairs: %d" % num_pairs)
print("#pixels: %d" % num_features)
print("#identities: %d" % num_classes)
# DOES X_train and X_test contain concatenated images??? (500, 250)
# Create training and test sets. 
X_train, X_test, t_train, t_test = train_test_split(images, t, test_size=0.25, random_state=0)

###############################################################################
#   NEURAL NETWORK CONSTANTS                                                  #
###############################################################################
# HYPERPARAMETERS
NUM_TRAINING_STEPS = 500

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


fully_connected_nodes = 100


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
_, w, h, d = pool2_h.get_shape().as_list()
pool2_h_flat = tf.reshape(pool2_h, [-1, w*h*d])


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
output_layer = tf.sigmoid(tf.matmul(fc1_h, out_w) + out_b)
print_output = tf.Print(output_layer, [output_layer], message='Output values : ')

print("output_layer.get_shape()")
print(output_layer.get_shape())

# Layer storing the actual target values.
target_layer = tf.placeholder(tf.float32, [None, num_classes])
#print_target = tf.Print(target_layer, [target_layer], message='Target values : ')

# The loss function.
cross_entropy = tf.reduce_mean(tf.square(output_layer - target_layer)) # tf.nn.sigmoid_cross_entropy_with_logits(output_layer, target_layer) # tf.reduce_mean(
print_loss = tf.Print(cross_entropy, [cross_entropy], message='Loss values : ')

# The training step.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


###############################################################################
#   CALCULATE ACCURACY                                                        #
###############################################################################
def accuracy(X, t):
    correct_prediction = tf.equal(tf.round(tf.to_float(output_layer), tf.to_float(target_layer)))
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

    _, loss, out = sess.run([train_step, print_output, print_loss], feed_dict={target_layer:t_batch, input_layer:X_batch})
    
    if i % 10 == 9:
        train_time = time() - t0
        print('Batch %d: %f seconds' % (i, train_time))
        t0=time()

 

###############################################################################
#   EVALUATE                                                                  #
###############################################################################
#print('TEST ACCURACY: ')
#print(accuracy(X_test, t_test))


###############################################################################
#   RUN THE NEURAL NETWORK                                                    #
###############################################################################
print("Predicting people's names on the test set")
test = tf.round(output_layer)
y_pred = sess.run(test, feed_dict={input_layer:X_test})
t_pred = t_test


print(classification_report(t_pred, y_pred))
print(confusion_matrix(t_pred, y_pred))
