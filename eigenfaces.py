#######################################################################
# EIGENFACES                                                          #
# ------------------------------------------------------------------- #
# Performs facial recognition on the LFW dataset using PCA and the    #
# eigenfaces algorithm.                                               #
#######################################################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier as knn


from pca import pca, transform
from time import time
import os
import sys
import mkl
mkl.set_num_threads(16)
import load_lfw
from time import time
perror = print


# Create a training function.
def create_train_test(k, classifier='logistic'):
	# Define a training function.
	def train(set, classes, pairs, targets):
		perror('Beginning eigenface training procedure.')
		# Create a training matrix.
		names, faces = zip(*set)
		faces = np.array(faces)
		n, w, h = faces.shape
		faces = faces.reshape([n, w * h])
		perror('Running on %d images with resolution %d x %d' % (n, w, h))

		# Perform PCA.
		perror('Performing principal component analysis with %d components.' % k)
		t0 = time()
		components = np.array(pca(faces, k))
		perror('Completed principal component analysis in %.3f seconds' % (time() - t0))

		# Transform the individual faces.
		Y = transform(faces, components, faces)

		# Transform the pairs of faces.
		P = np.array(pairs).reshape([-1, w * h])
		P = transform(P, components, faces)
		P = P.reshape([-1, k * 2])
		Q = np.zeros([len(pairs)], dtype=np.float64)
		for i in range(len(pairs)):
			Q[i] = np.linalg.norm(P[i][:k] - P[i][k:])
		P = Q.reshape(-1, 1)
		perror('Transformed all faces into the component space.')

		# Train a classifier.
		if classifier == 'logistic_pairs':
			t0 = time()
			perror('Performing logistic regression on pairs.')
			model = lr()
			model.fit(P, targets)
			perror('Logistic regression on pairs completed in %.3f seconds.' % (time() - t0))
		elif classifier == 'logistic':
			t0 = time()
			perror('Performing logistic regression.')
			model = lr(solver='lbfgs', multi_class='multinomial')
			model.fit(Y, names)
			perror('Logistic regression completed in %.3f seconds.' % (time() - t0))
		elif classifier == 'knn':
			t0 = time()
			perror('Performing knn classification.')
			model = knn(weights = 'distance')
			model.fit(Y, names)
			perror('KNN completed in %.3f seconds.' % (time() - t0))

		return model, components, faces

	def same_or_different(face1, face2, m):
		model, components, faces = m
		f1 = transform(np.ravel(face1).reshape(1, -1), components, faces)
		f2 = transform(np.ravel(face2).reshape(1, -1), components, faces)

		if classifier == 'logistic_pairs':
			return model.predict(np.array([np.linalg.norm(f1 - f2)]).reshape([1, -1]))[0]
		else:
			y1 = model.predict(transform(f1.reshape(1, -1), components, faces))[0]
			y2 = model.predict(transform(f2.reshape(1, -1), components, faces))[0]
			return y1 == y2

	def classify(face1, m):
		model, components, faces = m
		f1 = np.ravel(face1)
		return model.predict(transform(f1.reshape(1, -1), components, faces))[0]

	return train, same_or_different, classify

#train, test = create_train_test(50)
#print(load_lfw.run_test_unrestricted(1, train, test, type='funneled', resize=.3, color=False))

