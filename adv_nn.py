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
def generate_model(embedding, width, height, style='classification', color=False):
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
        print(num_names)
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
            input_layer = tf.placeholder(tf.float32, [1, width, height, depth])

            # First convolutional layer.
            conv1_w = weight_variable([5, 5, 1, 16])
            conv1_b = bias_variable([16])
            conv1_h = tf.nn.relu(conv(input_layer, conv1_w) + conv1_b)

            # Pooling layer.
            pool1_h = max_pool(conv1_h, w=2, h=2, sw=2, sh=2)
            print(pool1_h.get_shape())
            # Second convolutional layer.
            conv2_w = weight_variable([5, 5, 16, 32])
            conv2_b = bias_variable([32])
            conv2_h = tf.nn.relu(conv(pool1_h, conv2_w) + conv2_b)
            print(conv2_h.get_shape())

            # Pooling layer.
            pool2_h = max_pool(conv2_h, w=2, h=2, sw=1, sh=1)
            print(pool2_h.get_shape())
            _, w, h, d = pool2_h.get_shape().as_list()
            pool2_h_flat = tf.reshape(pool2_h, [-1, w*h*d])

            # Fully connected layer.
            fc1_w = weight_variable([w*h*d, 100])
            fc1_b = bias_variable([100])
            fc1_h = tf.nn.relu(tf.matmul(pool2_h_flat, fc1_w) + fc1_b)

            # Output layer.
            out_w = weight_variable([100, embedding])
            out_b = bias_variable([embedding])
            output_layer = tf.nn.l2_normalize(tf.matmul(fc1_h, out_w) + out_b, 1)

            # Classification layer.
            class_w = weight_variable([embedding, num_names])
            class_b = bias_variable([num_names])
            classification = tf.matmul(output_layer, class_w) + class_b

            correct_classification = tf.placeholder(tf.float32, [1, num_names])
            xentropy = tf.nn.softmax_cross_entropy_with_logits(classification, correct_classification)
            classification_loss = tf.reduce_mean(xentropy)
            train_classification = tf.train.AdamOptimizer(1e-4).minimize(classification_loss)
            classification_loss_summary = tf.scalar_summary('classification_loss', classification_loss)
            #classification_accuracy_summary = tf.scalar_summary('classification_accuracy', tf.cast(tf.equal(tf.argmax(correct_classification, 1), tf.argmax(classification, 1)), tf.int32))

            # Positive and negative neighboring images.
            #positive = tf.placeholder(tf.float32, [None, embedding])
            #negative = tf.placeholder(tf.float32, [None, embedding])
            #alpha = .2
            #loss1 = tf.nn.l2_loss(output_layer - positive) - tf.nn.l2_loss(output_layer - negative) + alpha
            #triplet_loss = tf.reduce_sum(loss1)
            #train_triplet = tf.train.AdamOptimizer(1e-4).minimize(triplet_loss)

            # Pairwise loss.
            #other = tf.placeholder(tf.float32, [1, embedding])
            #l2norm2 = tf.reduce_sum(tf.square(output_layer - other))
            #pairwise_loss_diff = tf.square(tf.nn.relu(tf.sub(tf.constant(.1), tf.sqrt(l2norm2))))
            #pairwise_loss_same = l2norm2

            # Pairwise training.
            #pairwise_train_same = tf.train.AdamOptimizer(1e-4).minimize(pairwise_loss_same)
            #pairwise_train_diff = tf.train.AdamOptimizer(1e-4).minimize(pairwise_loss_diff)

            # Pairwise summaries.
            #l2norm_sum_diff = tf.scalar_summary('l2norm-diff', l2norm2)
            #l2norm_sum_same = tf.scalar_summary('l2norm-same', l2norm2)
            #loss_sum_same = tf.scalar_summary('pairwise-loss-same', pairwise_loss_same)
            #loss_sum_diff = tf.scalar_summary('pairwise-loss-diff', pairwise_loss_diff)

            # Initialize the graph.
            sess.run(tf.initialize_all_variables())

        #####################################################################
        # Run pairwise training.                                            #
        #####################################################################
        if style=='classification':
            for i in range(len(pairs) * 10):
                p1, p2 = pairs[i % len(pairs)]
                n1, n2 = names[i % len(pairs)]

                for (p, n) in [[p1, n1], [p2, n2]]:
                    with graph.as_default():
                        feed = {}
                        feed[input_layer] = p.reshape([1, 50, 50, 1])
                        feed[correct_classification] = name_to_onehot(n)
                        _, a, b = sess.run([train_classification, classification_loss_summary, classification_loss_summary], feed_dict=feed)

                        writer.add_summary(a, i)
                        writer.add_summary(b, i)

            return sess, graph, input_layer, classification

        elif style=='pairwise':
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
            for i in range(len(pairs)):
                p1, p2 = pairs_np[i]
                #if targets_np[i]:
                #    continue

                # Run the first image through the network.
                with graph.as_default():
                    # Run the first image through the network.
                    inp1 = p1.reshape([1, width, height, 1])
                    fv1 = sess.run(output_layer, feed_dict={input_layer:inp1})

                    # Train the network based on the second image.
                    inp2 = p2.reshape([1, width, height, 1])
                    feed = {}
                    feed[input_layer] = inp2
                    feed[other] = fv1

                    # For tensorboard.
                    if targets_np[i] == True:
                        _, s, l = sess.run([pairwise_train_same, loss_sum_same, l2norm_sum_same], feed_dict=feed)
                    elif targets_np[i] == False:
                        _, s, l = sess.run([pairwise_train_diff, loss_sum_diff, l2norm_sum_diff], feed_dict=feed)
                    else:
                        raise Exception('Unexpected target value.')

                    writer.add_summary(s, i)
                    writer.add_summary(l, i)

                # Print useful status updates.
                if i % 600 == 0 and i != 0:
                    print('Trained on 600 pairs in %.3f seconds.' % (time() - t1))
                    t1 = time()

            print('Trained on %d pairs in %.3f seconds.' % (len(pairs), time() - t0))
            t0 = time()

            # Run all images through the network.
            print('Running all %d images through the network.' % (2 * len(pairs)))
            np_pairs = np.array(pairs).reshape([-1, 1, width, height, 1])
            embedded_pairs = np.zeros([np_pairs.shape[0], embedding])
            with graph.as_default():
                for i in range(np_pairs.shape[0]):
                    embedded_pairs[i] = sess.run(output_layer, feed_dict={input_layer:np_pairs[i]})
                #embedded_pairs = sess.run(output_layer, feed_dict={input_layer:np_pairs})
            print(embedded_pairs.shape)
            print('Ran all images through the network in %.3f seconds.' % (time() - t0))

            # Calculate the distances between pairs of images.
            t0 = time()
            print('Calculating distances between pairs.')
            distances = np.zeros([len(pairs), 1])
            for i in range(len(pairs)):
                distances[i][0] = np.linalg.norm(embedded_pairs[i] - embedded_pairs[i+1])
            print('Calculated distances in %.3f seconds.' % (time() - t0))

            # Perform logistic regression on the distances using the target values.
            print('Performing logistic regression.')
            lr = LogisticRegression()
            lr.fit(distances, targets)
            print('Logistic regression completed in %.3f seconds.' % (time() - t0))

            outcomes = lr.predict(distances)
            correct = 0
            for i in range(len(pairs)):
                if outcomes[i] == targets[i]:
                    correct += 1

            print('Error rate on training data: %f' % (correct / float(len(pairs))))
        #####################################################################
        # Ingest the images.                                                #
        #####################################################################
        #index = {}
        #for (ps, ns) in zip(pairs, names):
        #    for p, n in zip(ps, ns):
        #        if n not in index: index[n] = []
        #        index[n].append(p)

        ######################################################################
        # Run adversarial training.                                          #
        ######################################################################
        #for i in range(100):
        #    pass
            # Select a random bucket.
        # Run a quick test on the first pair.
        #p1, p2 = pairs[0]
        #p1, p2 = p1.reshape(width, height, 1), p2.reshape(width, height, 1)
        #result = sess.run(out, feed_dict={input_layer:np.array([p1, p2])})

        #print(result.shape)

        return sess, graph, input_layer, output_layer, lr

    def outcome_fn(face1, face2, model):
        sess, graph, input_layer, output_layer, lr = model

        # Run the faces through the network.
        with graph.as_default():
            faces = np.array([face1, face2]).reshape([2, width, height, 1])
            embeddings = sess.run(output_layer, {input_layer:faces})

        # Run the distance through the logistic regression.
        return lr.predict(np.linalg.norm(embeddings[0] - embeddings[1]))

    return train_fn, outcome_fn
