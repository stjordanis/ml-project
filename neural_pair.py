import random
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
def generate_model(embedding, width, height, iterations, batch_size=1, color=False, choices=None, randomized_pairs=True):
    if color:
        depth = 3
    else:
        depth = 1

    def train_fn(pairs, targets, names, ids):
        ######################################################################
        # Initial bookkeeping.                                               #
        ######################################################################
        # Index photos by ids.
        index = {}
        for i in range(len(pairs)):
            for j in range(2):
                index[ids[i][j]] = pairs[i][j].reshape([width, height, depth])
        index_keys = list(index.keys())

        # Determine the list of identities and index by identities.
        name_to_face = {}

        for ids2, ns in zip(ids, names):
            for (id, n) in zip(ids2, ns):
                if n not in name_to_face:
                    name_to_face[n] = []
                name_to_face[n].append(id)
        num_names = len(name_to_face.keys())
        num_faces = len(pairs)*2
        num_pairs = len(pairs)

        # Determine which faces have parntners.
        partners = {}
        has_partner = []
        for i in range(num_pairs):
            for j in range(2):
                name = names[i][j]
                partners[ids[i][j]] = list(name_to_face[name])
                partners[ids[i][j]].remove(ids[i][j])
                if len(name_to_face[name]) > 1:
                    has_partner.append(ids[i][j])

        ######################################################################
        # Build the neural net.                                              #
        ######################################################################
        graph = tf.Graph()
        sess = tf.Session(graph=graph)
        writer = tf.train.SummaryWriter('logs/%d' % time(), graph)

        with graph.as_default():
            # Input layer.
            input_layer1 = tf.placeholder(tf.float32, [None, width, height, depth])
            input_layer2 = tf.placeholder(tf.float32, [None, width, height, depth])

            # First convolutional layer.
            conv1_w = weight_variable([3, 3, depth, 16])
            conv1_b = bias_variable([16])
            conv1_h1 = tf.nn.relu(conv(input_layer1, conv1_w) + conv1_b)
            conv1_h2 = tf.nn.relu(conv(input_layer2, conv1_w) + conv1_b)

            # Pooling layer.
            pool1_h1 = max_pool(conv1_h1, w=3, h=3, sw=2, sh=2)
            pool1_h2 = max_pool(conv1_h2, w=3, h=3, sw=2, sh=2)

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
            fc1_w = weight_variable([w*h*d,75])
            fc1_b = bias_variable([75])
            fc1_h1 = tf.nn.relu(tf.matmul(pool2_h1_flat, fc1_w) + fc1_b)
            fc1_h2 = tf.nn.relu(tf.matmul(pool2_h2_flat, fc1_w) + fc1_b)

            # Output layer.
            out_w = weight_variable([75, embedding])
            out_b = bias_variable([embedding])
            output_layer1 = tf.nn.l2_normalize(tf.matmul(fc1_h1, out_w) + out_b, 1)
            output_layer2 = tf.nn.l2_normalize(tf.matmul(fc1_h2, out_w) + out_b, 1)
            distance_between = tf.reduce_sum(tf.square(output_layer1 - output_layer2), 1)

            # Pairwise loss.
            l2norm2 = tf.reduce_sum(tf.square(output_layer1 - output_layer2))
            target_inputs = tf.placeholder(tf.float32, [None])

            #loss_same = l2norm2
            #loss_diff = tf.square(tf.nn.relu(tf.sub(tf.constant(.2), tf.sqrt(l2norm2))))
            #loss = tf.reduce_mean(tf.cond(tf.equal(target_inputs, 1)
            loss = tf.reduce_mean(tf.mul(target_inputs, l2norm2))

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
        print('Beginning training for pairwise loss.')
        if randomized_pairs: print('Training on randomly generated pairs.')
        else: print('Training on pairs from LFW.')

        t0 = time()
        t1 = t0

        # Train.
        training_face1 = []
        training_face2 = []
        training_boolean = []

        for i in range(iterations):
            batch_face1, batch_face2, batch_target = [], [], []

            for j in range(batch_size):
                # If we're using random pairs.
                if randomized_pairs:
                    # 1) Decide whether we're going to pick the same person.
                    same = np.random.binomial(1, .5) == 1
                    training_boolean.append(same)
                    if same: batch_target.append(1)
                    else: batch_target.append(-1)
    
                    # 2) Decide who to pick.
                    if choices is not None:
                        face1, face2 = select_faces(same, partners, has_partner, index, index_keys, sess, graph, input_layer1, input_layer2, distance_between, choices=choices)
                    else:
                        if same:
                            id1 = has_partner[np.random.randint(0, len(has_partner))]
                            id2 = partners[id1][np.random.randint(0, len(partners[id1]))]
                        else:
                            id1 = index_keys[np.random.randint(0, len(index_keys))]
                            id2 = index_keys[np.random.randint(0, len(index_keys))]
                            while id2 in partners[id1]:
                                id2 = index_keys[np.random.randint(0, len(index_keys))]
                        face1, face2 = index[id1], index[id2]

                # If we're going to use the training pairs.
                else:
                    row = np.random.randint(0, len(pairs))
                    face1, face2 = index[ids[row][0]], index[ids[row][1]]
                    if targets[row]: batch_target.append(1)
                    else: batch_target.append(-1)
                    training_boolean.append(targets[row])

                # Build the proper arrays.
                training_face1.append(face1)
                training_face2.append(face2)
                batch_face1.append(face1)
                batch_face2.append(face2)

            # Feed dictionary.
            feed = {
                input_layer1:np.array(batch_face1),
                input_layer2:np.array(batch_face2),
                target_inputs:np.array(batch_target),
            }

            # Run the training step.
            sess.run([pairwise_train], feed_dict=feed)

            # Print useful status updates.
            if i % 1000 == 0 and i != 0:
                print('Trained on %d batches in %.3f seconds.' % (i, time() - t1))
                t1 = time()

        print('Done training in %.3f seconds.' % (time() - t0))
        t0 = time()

        # Run all images through the network.
        print('Running the %d training examples through the network.' % (iterations * batch_size))
        with graph.as_default():
            feed_dict = {input_layer1:np.array(training_face1), input_layer2:np.array(training_face2)}
            distances = sess.run(distance_between, feed_dict=feed_dict)
        print('Computed distances with the network in %.3f seconds.' % (time() - t0))

        # Calculate the average distance between same and different pairs.
        same_avg, diff_avg = 0, 0
        for i, b in enumerate(training_boolean):
            if b: same_avg += distances[i]
            else: diff_avg += distances[i]
        print('Same average: %.5f' % (same_avg / float(len(targets) / 2)))
        print('Diff average: %.5f' % (diff_avg / float(len(targets) / 2)))

        # Perform logistic regression on the distances using the target values.
        print('Performing logistic regression.')
        lr = LogisticRegression()
        lr.fit(distances.reshape(-1, 1), np.array(training_boolean))
        print('Logistic regression completed in %.3f seconds.' % (time() - t0))

        # Test the training data.
        outcomes = lr.predict(distances.reshape(-1, 1))
        correct = 0
        for i in range(iterations*batch_size):
            if outcomes[i] == training_boolean[i]: correct += 1
        print('Accuracy on training data: %.3f' % (correct / float(iterations*batch_size)))

        return sess, graph, input_layer1, input_layer2, distance_between, lr

    def outcome_fn(face1, face2, model):
        sess, graph, input_layer1, input_layer2, distance_between, lr = model

        # Run the faces through the network.
        with graph.as_default():
            face1 = face1.reshape([1, width, height, depth])
            face2 = face2.reshape([1, width, height, depth])
            distance = sess.run(distance_between, {input_layer1:face1, input_layer2:face2})

        # Run the distance through the logistic regression.
        return lr.predict(distance.reshape(-1, 1))[0]

    return train_fn, outcome_fn

def select_faces(same, partners, has_partner, index, index_keys, sess, graph, input_layer1, input_layer2, distance, choices=5):
    # Helpers.
    def random_index():
        k = np.random.randint(0, len(index_keys))
        return index_keys[k]

    #  Decide on the first person to use. If we're picking
    #  the same person, loop until we find a face that has
    #  a pair in the training set.
    if same:
        id1 = has_partner[np.random.randint(0, len(has_partner))]
    else:
        id1 = random_index()

    # If we're the same.
    if same: 
        random.shuffle(partners[id1])
        candidates = partners[id1][:choices]

    # If we're different.
    else:
        candidates = []
        while len(candidates) < choices:
            id2 = random_index()
            if id2 not in partners[id1] and id2 not in candidates:
                candidates.append(id2)

    # Run each of the candidates through the network.
    face1 = [index[id1]] * len(candidates)
    candidate_faces = []
    for id in candidates: candidate_faces.append(index[id])

    with graph.as_default():
        distances = sess.run(distance, feed_dict={input_layer1:np.array(candidate_faces), input_layer2:np.array(face1)})

    # Return the hardest choice.
    if same: id2 = candidates[np.argmax(distances)]
    else:    id2 = candidates[np.argmin(distances)]

    return index[id1], index[id2]
