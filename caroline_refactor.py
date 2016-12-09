from time import time

#from sklearn.model_selection import train_test_split
#from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys
import tensorflow as tf
import numpy as np
import load_lfw


###############################################################################
#   NEURAL NETWORK CONSTANTS                                                  #
###############################################################################
# HYPERPARAMETERS
NUM_TRAINING_STEPS = 200

batch_size = 30
learning_rate = 1e-4
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


fully_connected_nodes = 30

#############################################################################
# HELPER FUNCTIONS                                                          #
#############################################################################

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

##############################################################################
# TRAINING FUNCTION                                                          #
##############################################################################
def generate_model(embedding, width, height, iterations, batch_size=1, style='concatenated', color=False): #TODO change pairwise

	def train_fn(pairs, targets, names):
    	# TODO initial bookkeeping
    	######################################################################
        # Build the neural net.                                              #
        ######################################################################
		graph = tf.Graph()
		sess = tf.Session(graph=graph)
		writer = tf.train.SummaryWriter('logs/%d' % time(), graph)

		if color:
			depth = 3
		else:
			depth = 1

		print("DATA FORMATTING")
		# DATA FORMATTING
		training_faces = []
		# concatenate pairs
		for pair in pairs:
			#print(pair[0].shape)
			#print(pair[1].shape)
			#print("width = ", width)
			#print("height = ", height)
			#print("depth = ", depth)
			next_pair = np.concatenate((pair[0].reshape([width, height, depth]), pair[1].reshape([width, height, depth])), axis=0)
			training_faces.append(next_pair)
		training_faces = np.asarray(training_faces)
		print("training_faces.shape = ", training_faces.shape)
		print("DONE CONCATENATING FACES")

		with graph.as_default():

			# Input layer.
			input_layer = tf.placeholder(tf.float32, [None, 2 * width, height, depth])

			#print("input_layer.get_shape():")
			#print(input_layer.get_shape())

			# Convolutional layer 1
			conv1_w = weight_variable([layer1_filter_size, layer1_filter_size, depth, layer1_depth]) #depth or 1?
			conv1_b = bias_variable([layer1_depth])
			conv1_h = tf.nn.relu(conv(input_layer, conv1_w) + conv1_b)

			#print("conv1_h.get_shape()")
			#print(conv1_h.get_shape())

			# Pooling layer 1
			pool1_h = max_pool(conv1_h, w=2, h=2, sw=1, sh=1)

			#print("h_pool1.get_shape()")
			#print(h_pool1.get_shape())

			# Convolutional layer.
			conv2_w = weight_variable([layer2_filter_size, layer2_filter_size, layer1_depth, layer2_depth])
			conv2_b = bias_variable([layer2_depth])
			conv2_h = tf.nn.relu(conv(pool1_h, conv2_w) + conv2_b)

			#print("conv2_h.get_shape()")
			#print(conv2_h.get_shape())

			# Pooling layer.
			pool2_h = max_pool(conv2_h, w=2, h=2, sw=1, sh=1)
			_, w, h, d = pool2_h.get_shape().as_list()
			#print("d = ", d)
			pool2_h_flat = tf.reshape(pool2_h, [-1, w*h*d])


			#print("pool2_h_flat.get_shape()")
			#print(pool2_h_flat.get_shape())

			# Fully connected layer.
			fc1_w = weight_variable([w*h*d, fully_connected_nodes])
			fc1_b = bias_variable([fully_connected_nodes])
			fc1_h = tf.nn.relu(tf.matmul(pool2_h_flat, fc1_w) + fc1_b)

			#print("fc1_h.get_shape()")
			#print(fc1_h.get_shape())

			# Output layer.
			out_w = weight_variable([fully_connected_nodes, 1])
			out_b = bias_variable([1])
			output_layer = tf.sigmoid(tf.matmul(fc1_h, out_w) + out_b)
			print_output = tf.Print(output_layer, [output_layer], message='Output values : ')

			#print("output_layer.get_shape()")
			#print(output_layer.get_shape())


			# TODO loss functions?
			# Layer storing the actual target values.

			target_layer = tf.placeholder(tf.float32, [None, 1])
			print_target = tf.Print(target_layer, [target_layer], message='Target values : ')


			# The loss function.
			loss_func =  tf.reduce_mean(tf.square(output_layer - target_layer)) # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(output_layer, target_layer)) # 
			print_loss = tf.Print(loss_func, [loss_func], message='Loss values : ')

			# Initialize the graph.
			train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
			sess.run(tf.initialize_all_variables())

            # Run the training 
			targets = np.asarray(targets)
			targets = targets.reshape([5400, 1])

			for i in range(iterations):
			# The training step.
				print("i = ", i)
				index = np.random.randint(0,5400)
				_, out, loss, target  = sess.run([train_step, print_output, print_loss, print_target], feed_dict={target_layer:targets[index].reshape([1,1]), input_layer:training_faces[index].reshape([1, 50, 25, 1])})
				#_, loss, out = sess.run([train_step, print_output, print_loss], feed_dict={target_layer:targets, input_layer:training_faces})
				#_ = sess.run([train_step], feed_dict={target_layer:targets[i].reshape([1,1]), input_layer:training_faces[i].reshape([1, 50, 25, 1])})
				#_, loss, out = sess.run([train_step, print_output, print_loss], feed_dict={target_layer:targets, input_layer:training_faces})
			return sess, graph, input_layer, output_layer 

	def outcome_fn(face1, face2, model):
		if color:
			depth = 3
		else:
			depth = 1
		sess, graph, input_layer, output_layer = model
		
		# Run the faces through the network.
		with graph.as_default():

			next_input = concatenate_faces(face1, face2, depth)
			rounded_prediction = tf.round(output_layer)
			out = sess.run(rounded_prediction , feed_dict={input_layer: next_input}) #TODO

		# Run the prediction from the trained network
		return out

	def concatenate_faces(face1, face2, depth):
		next_pair = np.concatenate((face1.reshape([1, width, height, depth]), face2.reshape([1, width, height, depth])), axis=1)
		return next_pair

	return train_fn, outcome_fn





