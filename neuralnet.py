from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import numpy as np

##############################################################################
#   DATA WRANGLING                                                           #
##############################################################################
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)

# Find the shapes of the images.
num_images, h, w = lfw_people.images.shape

# Extract the input to the neural network and the number of features.
X = lfw_people.data
num_features = X.shape[1]

# Extract hte target data.
target_names = lfw_people.target_names
num_classes = target_names.shape[0]
t = np.zeros([num_images, num_classes])
t[np.arange(num_images), lfw_people.target] = 1

print("Total dataset size:")
print("#images: %d" % num_images)
print("#pixels: %d" % num_features)
print("#identities: %d" % num_classes)

# Create training and test sets.
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=0)

###############################################################################
#   NEURAL NETWORK CONSTANTS                                                  #
###############################################################################
batch_size = 20

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(.1, shape=shape)
    return tf.Variable(initial)

def conv(x, W, sw=1, sh=1):
    return tf.nn.conv2d(x, W, strides=[sw, sh, 1, 1], padding='SAME')

def max_pool(x, w=2, h=2, sw=2, sh=2):
    return tf.nn.max_pool(x, ksize=[1, w, h, 1], strides=[1, sw, sh, 1], padding='SAME')

###############################################################################
#   BUILD THE NEURAL NETWORK                                                  #
###############################################################################
# Input layer.
input_layer = tf.placeholder(tf.float32, [None, w * h])
input_layer_2d = tf.reshape(input_layer, [-1, w, h, 1])

print("input_layer_2d.get_shape():")
print(input_layer_2d.get_shape())

# Convolutional layer.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv(input_layer_2d, W_conv1) + b_conv1)

print("h_conv1.get_shape()")
print(h_conv1.get_shape())

# Pooling layer.
h_pool1 = max_pool(h_conv1, w=2, h=2, sw=2, sh=2)

print("h_pool1.get_shape()")
print(h_pool1.get_shape())

# Convolutional layer.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv(h_pool1, W_conv2) + b_conv2)

print("h_conv2.get_shape()")
print(h_conv2.get_shape())

# Pooling layer.
h_pool2 = max_pool(h_conv2, w=2, h=2, sw=2, sh=2)
_, pool2w, pool2h, pool2d = h_pool2.get_shape().as_list()
units = pool2w * pool2h * pool2d
h_pool2_flat = tf.reshape(h_pool2, [-1, units])

print("h_pool2_flat.get_shape()")
print(h_pool2_flat.get_shape())

# Fully connected layer.
W_fc1 = weight_variable([units, 100])
b_fc1 = bias_variable([100])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

print("h_fc1.get_shape()")
print(h_fc1.get_shape())

# Output layer.
W_output = weight_variable([100, num_classes])
b_output = bias_variable([num_classes])
output_layer = tf.matmul(h_fc1, W_output) + b_output

print("output_layer.get_shape()")
print(output_layer.get_shape())

# Layer storing the actual target values.
target_layer = tf.placeholder(tf.float32, [None, num_classes])

# The loss function.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer, target_layer))

# The training step.
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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
for i in range(1000):
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
