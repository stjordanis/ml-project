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

def perror(s):
    pass
def perror2(s):
    sys.stderr.write(s + '\n')

##############################################################################
# TRAINING FUNCTION                                                          #
##############################################################################
def generate_model(embedding, width, height, iterations, batch_size=1, color=False, consider=None, num_lr=5000, roc=False):
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

        # Update choices.
        choices = consider
        if consider is None: choices = 1

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
            conv1_w = weight_variable([3, 3, depth, 32])
            conv1_b = bias_variable([32])
            conv1_h1 = tf.nn.relu(conv(input_layer1, conv1_w) + conv1_b)
            conv1_h2 = tf.nn.relu(conv(input_layer2, conv1_w) + conv1_b)

            # Pooling layer.
            pool1_h1 = max_pool(conv1_h1, w=3, h=3, sw=2, sh=2)
            pool1_h2 = max_pool(conv1_h2, w=3, h=3, sw=2, sh=2)

            # Second convolutional layer.
            conv2_w = weight_variable([3, 3, 32, 64])
            conv2_b = bias_variable([64])
            conv2_h1 = tf.nn.relu(conv(pool1_h1, conv2_w) + conv2_b)
            conv2_h2 = tf.nn.relu(conv(pool1_h2, conv2_w) + conv2_b)

            # Pooling layer.
            pool2_h1 = max_pool(conv2_h1, w=3, h=3, sw=1, sh=1)
            pool2_h2 = max_pool(conv2_h2, w=3, h=3, sw=1, sh=1)
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
            target_inputs = tf.placeholder(tf.float32, [None])
            output_layer1 = tf.nn.l2_normalize(tf.matmul(fc1_h1, out_w) + out_b, 1)
            output_layer2 = tf.nn.l2_normalize(tf.matmul(fc1_h2, out_w) + out_b, 1)

            dists = tf.reduce_sum(tf.square(output_layer1 - output_layer2), 1)
            l2norm2 = tf.reduce_sum(tf.square(output_layer1 - output_layer2))
            loss = tf.reduce_mean(tf.mul(target_inputs, dists))

            # Pairwise training.
            pair_train = tf.train.AdamOptimizer(1e-4).minimize(loss)

            # Pairwise summaries.
            l2norm_diff = tf.scalar_summary('l2norm-diff', l2norm2)
            l2norm_same = tf.scalar_summary('l2norm-same', l2norm2)
            loss_same = tf.scalar_summary('loss-same', loss)
            loss_diff = tf.scalar_summary('loss-diff', loss)

            # Initialize the graph.
            sess.run(tf.initialize_all_variables())

        #####################################################################
        # Run pairwise training.                                            #
        #####################################################################
        perror('Beginning training for pair loss.')
        perror('Training on randomly generated triplets.')

        t0 = time()
        t1 = t0

        # Train.
        for i in range(iterations):
            batch_f1, batch_f2, batch_t = [], [], []

            for j in range(batch_size):
                same = np.random.binomial(1, .5) == 1

                # Pick a pair of identical faces.
                if same:
                    anch = has_partner[np.random.randint(0, len(has_partner))]
                    random.shuffle(partners[anch])
                    candidates = partners[anch][:choices]
                    batch_t.append(1)

                # Pick a different face.
                else:
                    anch = index_keys[np.random.randint(0, len(index_keys))]
                    candidates = []
                    while len(candidates) < choices:
                        diff = None
                        while diff is None or diff == anch or diff in partners[anch]:
                            diff = index_keys[np.random.randint(0, len(index_keys))]
                        candidates.append(diff)
                    batch_t.append(-1)

                # Get the faces.
                anchF = index[anch]
                f2 = []
                for f in candidates: f2.append(index[f])

                # Decide on the best face to use.
                def choose_face(candidates, input_layer, arg):
                    if len(candidates) == 1: return candidates[0]
                    ds = sess.run(l2norm2, {input_layer2:np.array([anchF] * len(candidates)), input_layer:np.array(candidates)})
                    return candidates[arg(ds)]

                if not same:
                    a_sum, b_sum = l2norm_diff, loss_diff
                    f2 = choose_face(f2, input_layer1, np.argmin)
                else:
                    a_sum, b_sum = l2norm_same, loss_same
                    f2 = choose_face(f2, input_layer1, np.argmax)

                # Build the proper arrays.
                batch_f1.append(anchF)
                batch_f2.append(f2)

            # Feed dictionary.
            feed = {
                input_layer1:np.array(batch_f2),
                input_layer2:np.array(batch_f1),
                target_inputs:np.array(batch_t)
            }

            # Run the training step.
            _, a, b = sess.run([pair_train, a_sum, b_sum], feed_dict=feed)
            writer.add_summary(a, i)
            writer.add_summary(b, i)

            # Print useful status updates.
            if i % 1000 == 0 and i != 0:
                perror('Trained on %d batches in %.3f seconds.' % (i, time() - t1))
                t1 = time()

        perror2('Done training in %.3f seconds.' % (time() - t0))
        t0 = time()

        # Run randomly generated faces through the network.
        faces1, faces2, bools = [], [], []
        for i in range(num_lr):
            same = np.random.binomial(1, .5) == 1
            if same:
                f1 = has_partner[np.random.randint(0, len(has_partner))]
                random.shuffle(partners[f1])
                f2 = partners[f1][0]
            else:
                f1 = index_keys[np.random.randint(0, len(index_keys))]
                f2 = None
                while f2 is None or f2 == f1 or f2 in partners[f1]:
                    f2 = index_keys[np.random.randint(0, len(index_keys))]
            faces1.append(index[f1])
            faces2.append(index[f2])
            bools.append(same)

        perror('Running the %d training examples through the network.' % (iterations * batch_size))
        with graph.as_default():
            feed_dict = {input_layer1:np.array(faces1), input_layer2:np.array(faces2)}
            distances = sess.run(dists, feed_dict=feed_dict)
        perror('Computed distances with the network in %.3f seconds.' % (time() - t0))

        # Perform logistic regression on the distances using the target values.
        perror('Performing logistic regression.')
        lr = LogisticRegression()
        lr.fit(distances.reshape(-1, 1), np.array(bools))
        perror('Logistic regression completed in %.3f seconds.' % (time() - t0))

        return sess, graph, input_layer1, input_layer2, dists, lr

    def outcome_fn(face1, face2, model):
        sess, graph, input_layer1, input_layer2, distance_same, lr = model

        # Run the faces through the network.
        with graph.as_default():
            face1 = face1.reshape([1, width, height, depth])
            face2 = face2.reshape([1, width, height, depth])
            distance = sess.run(distance_same, {input_layer1:face1, input_layer2:face2})

        # Run the distance through the logistic regression.
        pred = lr.predict(distance.reshape(-1, 1))[0]
        if roc:
            return distance, pred
        else:
            return pred

    return train_fn, outcome_fn
