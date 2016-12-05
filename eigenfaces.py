#######################################################################
# EIGENFACES                                                          #
# ------------------------------------------------------------------- #
# Performs facial recognition on the LFW dataset using PCA and the    #
# eigenfaces algorithm.                                               #
#######################################################################
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression as lr


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
def instance(k, classifier='logistic', dim_reduction='pca', feature='distance'):
	# Define a training function.
	def train(pairs, targets, names):
		# Create a training matrix.
		perror('Beginning eigenface training procedure.')
		n = len(pairs)
		perror(len(pairs))
		w, h = pairs[0][0].shape[:2]
		perror('Running on %d pairs of images with resolution %d x %d' % (n, w, h))
		perror('Overall shape of each image: %s' % (str(pairs[0][0].shape)))
		# Prepare to perofrm dimensionality reduction.
		P = np.array(pairs).reshape([2*n, -1])
		N = np.ravel(np.array(names))
		perror('Shape of numpy array of images: %s' % str(P.shape))

		# Create a transformer.
		t0 = time()
		perror('Finding the basis for dimensionality reduction.')
		if dim_reduction == 'fisher':
			transformer = fisher_transformer(P, N, k)
		elif dim_reduction == 'pca':
			transformer = pca_transformer(P, k, save=False)
		elif dim_reduction.startswith('fisher_pca'):
			pca_k = int(dim_reduction.rsplit('_', 1)[1])
			transformer = fisher_transformer(P, N, k, pca_first=pca_k)
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


# Some tests to try:
# 1) Push PCA with a variable number of components, cranking up the resolution.
# 2) Try cropping. You can crop down to (115, 250) and still get decent results.
# 3) Try with Fisher alone. Fisher seems to benefit from small feature spaces. Try with resize=.1
# 4) Try PCA followed by Fisher. 100 components at resize=.3 and crop=(115, 250) seemed to get about
#    75% accuracy on the first fold of LFW.
# 5) Best results: Color=True, resize=.15, dim_reduction=fisher_pca_100, k=5
#train, test = create_train_test(50)
#print(load_lfw.run_test_unrestricted(1, train, test, type='funneled', resize=.3, color=False))

