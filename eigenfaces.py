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


from pca import pca_transformer, fisher_transformer
from time import time
import os
import sys
import mkl
mkl.set_num_threads(16)
import load_lfw
from time import time
perror = print


# Create a training function.
def instance(k, classifier='logistic', fisher=False, feature='pair'):
	# Define a training function.
	def train(pairs, targets, names):
		# Create a training matrix.
		perror('Beginning eigenface training procedure.')
		n = len(pairs)
		perror(len(pairs))
		w, h = pairs[0][0].shape[:2]
		perror('Running on %d pairs of images with resolution %d x %d' % (n, w, h))

		# Prepare to perofrm dimensionality reduction.
		P = np.array(pairs).reshape([2*n, -1])
		N = np.ravel(np.array(names))

		# Create a transformer.
		t0 = time()
		perror('Finding the basis for dimensionality reduction.')
		if fisher:
			transformer = fisher_transformer(P, N, k)
		else:
			transformer = pca_transformer(P, k, save=False)
		perror('Found the basis in %.3f seconds' % (time() - t0))

		# Transform the training data and put it back into pairs.
		t0 = time()
		P = transformer.transform(P).reshape([-1, k * 2])

		# If distances were requested, calculate distances.
		if feature=='distance':
			Q = np.zeros(n)
			for i in range(n):
				Q[i] = np.linalg.norm(P[i][:k] - P[i][k:])
			P = Q.reshape(-1, 1)
		elif feature=='difference':
			Q = np.zeros((n, k))
			for i in range(n):
				Q[i] = P[i][:k] - P[i][k:]
			P = Q
		perror('Transformed all faces into the component space in %.3f seconds.' % (time() - t0))


		# Build the appropriate classifier.
		if classifier == 'logistic':
			t0 = time()
			perror('Performing logistic regression.')
			model = lr()
			model.fit(P, targets)
			perror('Logistic regression completed in %.3f seconds.' % (time() - t0))
		elif classifier == 'distance':
			def model(fv1, fv2):
				return np.linalg.norm(fv1 - fv2) < 10


		return model, transformer

	def same_or_different(face1, face2, m):
		model, transformer = m

		# Transform the faces.
		f1 = transformer.transform(np.ravel(face1))
		f2 = transformer.transform(np.ravel(face2))

		if classifier == 'distance':
			return model(f1, f2)

		if feature=='distance':
			feature_vec = np.linalg.norm(f1 - f2)
		elif feature=='difference':
			feature_vec = f1 - f2
		else:
			feature_vec = np.ravel(np.array([f1, f2]))
		feature_vec = feature_vec.reshape([1, -1])

		return model.predict(feature_vec)[0]

	return train, same_or_different

#train, test = create_train_test(50)
#print(load_lfw.run_test_unrestricted(1, train, test, type='funneled', resize=.3, color=False))

