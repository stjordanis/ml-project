from time import time
import tensorflow as tf
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression

#############################################################################
# HELPER FUNCTIONS                                                          #
#############################################################################
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

##############################################################################
# TRAINING FUNCTION                                                          #
##############################################################################
def generate_model(embedding, width, height, style='pairwise', color=False):
    def train_fn(pairs, targets, names):
        ######################################################################
        # Initial bookkeeping.                                               #
        ######################################################################
        # Determine the list of identities and index by identities.
        by_name = {}
        name_to_id = {}
        x = 0
        for ps, ns in zip(pairs, names):
            for (p, n) in zip(ps, ns):
                if n not in by_name:
                    by_name[n] = []
                    name_to_id[n] = x
                    x += 1
                by_name[n].append(p)
        num_names = len(by_name.keys())

        def name_to_onehot(name):
            a = np.zeros([1, num_names])
            a[0][name_to_id[name]] = 1
            return a

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

        with graph.as_default():

            # Input layer.
            input_layer1 = tf.placeholder(tf.float32, [None, width, height, depth])
            input_layer2 = tf.placeholder(tf.float32, [None, width, height, depth])

            # First convolutional layer.
            conv1_w = weight_variable([5, 5, 1, 16])
            conv1_b = bias_variable([16])
            conv1_h1 = tf.nn.relu(conv(input_layer1, conv1_w) + conv1_b)
            conv1_h2 = tf.nn.relu(conv(input_layer2, conv1_w) + conv1_b)

            # Pooling layer.
            pool1_h1 = max_pool(conv1_h1, w=2, h=2, sw=2, sh=2)
            pool1_h2 = max_pool(conv1_h2, w=2, h=2, sw=2, sh=2)

            # Second convolutional layer.
            conv2_w = weight_variable([5, 5, 16, 32])
            conv2_b = bias_variable([32])
            conv2_h1 = tf.nn.relu(conv(pool1_h1, conv2_w) + conv2_b)
            conv2_h2 = tf.nn.relu(conv(pool1_h2, conv2_w) + conv2_b)

            # Pooling layer.
            pool2_h1 = max_pool(conv2_h1, w=2, h=2, sw=1, sh=1)
            pool2_h2 = max_pool(conv2_h2, w=2, h=2, sw=1, sh=1)
            _, w, h, d = pool2_h1.get_shape().as_list()
            pool2_h1_flat = tf.reshape(pool2_h1, [-1, w*h*d])
            pool2_h2_flat = tf.reshape(pool2_h2, [-1, w*h*d])

            # Fully connected layer.
            fc1_w = weight_variable([w*h*d, 100])
            fc1_b = bias_variable([100])
            fc1_h1 = tf.nn.relu(tf.matmul(pool2_h1_flat, fc1_w) + fc1_b)
            fc1_h2 = tf.nn.relu(tf.matmul(pool2_h2_flat, fc1_w) + fc1_b)

            # Output layer.
            out_w = weight_variable([100, embedding])
            out_b = bias_variable([embedding])
            output_layer1 = tf.nn.l2_normalize(tf.matmul(fc1_h1, out_w) + out_b, 1)
            output_layer2 = tf.nn.l2_normalize(tf.matmul(fc1_h2, out_w) + out_b, 1)
            distance_between = tf.reduce_sum(tf.square(output_layer1 - output_layer2), 1)

            # Pairwise loss.
            l2norm2 = tf.reduce_sum(tf.square(output_layer1 - output_layer2))
            target_inputs = tf.placeholder(tf.float32, [None])
            loss = tf.reduce_mean(tf.mul(target_inputs, l2norm2))
            #pairwise_loss_diff = -l2norm2 #tf.square(tf.nn.relu(tf.sub(tf.constant(.1), tf.sqrt(l2norm2))))

            # Pairwise training.
            pairwise_train = tf.train.AdamOptimizer(1e-4).minimize(loss)

            # Pairwise summaries.
            l2norm_sum_diff = tf.scalar_summary('l2norm-diff', l2norm2)
            l2norm_sum_same = tf.scalar_summary('l2norm-same', l2norm2)
            loss_sum_same = tf.scalar_summary('pairwise-loss-same', loss) #pairwise_loss_same)
            loss_sum_diff = tf.scalar_summary('pairwise-loss-diff', loss) #pairwise_loss_diff)

            # Initialize the graph.
            sess.run(tf.initialize_all_variables())

        #####################################################################
        # Run pairwise training.                                            #
        #####################################################################
        if style=='pairwise':
            print('Beginning training for pairwise loss.')
            t0 = time()
            t1 = t0

            # Permute the data in various ways.
            pairs_rev = []
            for (a, b) in pairs:
                pairs_rev.append((b, a))
            pairs = list(pairs) + pairs_rev
            targets = list(targets) + list(targets)

            # Randomize the order of the examples.
            pairs_np = np.array(pairs)
            targets_np = np.array(targets)
            perm = np.random.permutation(len(pairs))
            pairs_np = pairs_np[perm]
            targets_np = targets_np[perm]
            pos = 0
            neg = 0

            # Train on every pair.
            for i in range(len(pairs)*2):
                j = i % len(pairs)
                p1, p2 = pairs_np[j]

                # Run the first image through the network.
                with graph.as_default():
                    # Run the first image through the network.
                    inp1 = p1.reshape([1, width, height, 1])
                    inp2 = p2.reshape([1, width, height, 1])

                    # Train the network based on the second image.
                    feed = {}
                    feed[input_layer1] = inp1
                    feed[input_layer2] = inp2
                    if targets_np[j]: feed[target_inputs] = np.array([1])
                    else: feed[target_inputs] = np.array([-1])

                    # For tensorboard.
                    if targets_np[j] == True:
                        _, s, l = sess.run([pairwise_train, loss_sum_same, l2norm_sum_same], feed_dict=feed)
                    else:
                        _, s, l = sess.run([pairwise_train, loss_sum_diff, l2norm_sum_diff], feed_dict=feed)

                    writer.add_summary(s, i)
                    writer.add_summary(l, i)

                # Print useful status updates.
                if i % 600 == 0 and i != 0:
                    print('Trained on %d pairs in %.3f seconds.' % (i, time() - t1))
                    t1 = time()

            print('Trained on %d pairs in %.3f seconds.' % (len(pairs), time() - t0))
            t0 = time()

            # Run all images through the network.
            print('Running all %d images through the network.' % (2 * len(pairs)))
            p1s, p2s = zip(*pairs)
            p1s = np.array(p1s).reshape([len(pairs), width, height, 1])
            p2s = np.array(p2s).reshape([len(pairs), width, height, 1])
            with graph.as_default():
                distances = sess.run(distance_between, feed_dict={input_layer1:p1s, input_layer2:p2s})
            print('Computed distances with the network in %.3f seconds.' % (time() - t0))

            # Calculate the average distance between same and different pairs.
            same_avg = 0
            diff_avg = 0
            for i in range(len(targets)):
                if targets[i]:
                    same_avg += distances[i]
                else:
                    diff_avg += distances[i]

            print('Same average: %.5f' % (same_avg / float(len(targets) / 2)))
            print('Diff average: %.5f' % (diff_avg / float(len(targets) / 2)))

            # Perform logistic regression on the distances using the target values.
            print('Performing logistic regression.')
            lr = LogisticRegression()
            lr.fit(distances.reshape(-1, 1), targets)
            print('Logistic regression completed in %.3f seconds.' % (time() - t0))

            outcomes = lr.predict(distances.reshape(-1, 1))
            correct = 0
            for i in range(len(pairs)):
                if outcomes[i] == targets[i]:
                    correct += 1

            print('Error rate on training data: %f' % (correct / float(len(pairs))))

        return sess, graph, input_layer1, input_layer2, distance_between, lr

    def outcome_fn(face1, face2, model):
        sess, graph, input_layer1, input_layer2, distance_between, lr = model

        # Run the faces through the network.
        with graph.as_default():
            face1 = face1.reshape([1, width, height, 1])
            face2 = face2.reshape([1, width, height, 1])
            distance = sess.run(distance_between, {input_layer1:face1, input_layer2:face2})

        # Run the distance through the logistic regression.
        return lr.predict(distance.reshape(-1, 1))[0]

    return train_fn, outcome_fn
