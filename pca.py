#######################################################################
# PRINCIPAL COMPONENTS ANALYSIS                                       #
# ------------------------------------------------------------------- #
# Given an (n x d) matrix where each row represents a feature vector  #
# of length d, performs principal components analysis, providing a    #
# basis that can be used for dimensionality reduction.                #
# This implementation was tested against the sklearn version on small #
# examples and arrived at the same answers.                           #
# ------------------------------------------------------------------- #
# Based on https://www.cs.unc.edu/~coombe/research/phd/svd.pdf        #
#######################################################################

import numpy as np
from sklearn.utils.extmath import fast_dot
from time import time
perror = print
import scipy.misc

#######################################################################
# A PCA TRANSFORMER                                                   #
#######################################################################
class pca_transformer:
	# Assumes that the feature vectors x(i) are in rows, and each
	# column represents a specific feature. There are n feature
	# vectors and d features in each vector.
	def __init__(self, X, num_components, save=False):
		# Basic initialization.
		n, self.d = X.shape
		components = []

		# Calculate the covariance matrix.
		self.mu = np.mean(X, axis=0)
		Z = X - self.mu
		A = fast_dot(Z.T, Z) / self.d

		# Perform PCA.
		i = 0
		while len(components) < num_components:
			eigenvector, eigenvalue = next_eigenvector(A)
			components.append(eigenvector)
			A = deflate(A, eigenvector, eigenvalue)
			if save:
				i += 1
				scipy.misc.imsave(str(i) + '.jpg', eigenvector.reshape([75, 75]))

		self.components = np.array(components)

	# X should be in the same form of above, or a singleton of length d.
	def transform(self, X):
		# Handle singleton inputs.
		if len(X.shape) == 1:
			X = X.reshape([1, -1])

		# Transform.
		B = transform(X, self.components, self.mu)

		# Handle singleton outputs.
		if len(X.shape) == 1:
			return B[0]
		return B

#######################################################################
# A FISHER TRANSFORMER                                                #
#######################################################################
class fisher_transformer:
	# Same as for PCA. {y} contains the corresponding classes for
	# each value of {X}.
	def __init__(self, X, y, num_components):
		# Basic initialization.
		n, d = X.shape
		components = []

		# Calculate the mean face.
		self.mu = np.mean(X, axis=0)

		# Determine the class means.
		classes = {}
		means = {}
		for i in range(n):
			# Handle new classes.
			if y[i] not in classes:
				classes[y[i]] = 0.0
				means[y[i]] = np.zeros([d])

			# Update counts and sums.
			classes[y[i]] += 1
			means[y[i]] += X[i]

		for c, mean in means.items():
			means[c] /= classes[c]


		# Calculate a centered matrix.
		Z = np.zeros(X.shape)
		for i in range(n):
			Z[i] = X[i] - means[y[i]]

		# Calculate the within-class scatter.
		t0 = time()
		perror('Calculating the within-class scatter.')
		Sw = fast_dot(Z.T, Z)
		perror('Calculated in %.3f seconds' % (time() - t0))

		# Calculate the between-class scatter.
		t0 = time()
		perror('Calculating the between-class scatter.')
		Z = np.zeros([len(classes.keys()), d])
		for i, (c, m) in enumerate(means.items()):
			Z[i] = np.sqrt(classes[c]) * (m - self.mu)
		Sb = fast_dot(Z.T, Z)
		perror('Calculated in %.3f seconds.' % (time() - t0))

		# The matrix whose eigenvectors we want.
		t0 = time()
		A = fast_dot(np.linalg.inv(Sw), Sb)
		perror('Inverted Sw and multiplied it by Sb in %.3f seconds.' % (time() - t0))

		# Find the top eigenvectors.
		while len(components) < num_components:
			eigenvector, eigenvalue = next_eigenvector(A)
			components.append(eigenvector)
			A = deflate(A, eigenvector, eigenvalue)

		self.components = np.array(components)

	# X should be in the same form of above, or a singleton of length d.
	def transform(self, X):
		# Handle singleton inputs.
		if len(X.shape) == 1:
			X = X.reshape([1, -1])

		# Transform.
		B = transform(X, self.components, self.mu)

		# Handle singleton outputs.
		if len(X.shape) == 1:
			return B[0]
		return B

#######################################################################
# HELPER FUNCTIONS SHARED BETWEEN PCA AND FISHER                      #
#######################################################################

# Projects the matrix X (same as in pca) onto the coordinate system
# specified by components.
def transform(X, components, mu):
    Z = X - mu
    return fast_dot(Z, components.T)

# Use the power method to calculate the eigenvector corresponding
# to the largest eigenvalue of X.
def next_eigenvector(A):
    n, d = A.shape

    # Initialze the eigenvector randomly.
    u = np.random.random([d])
    u = u / np.linalg.norm(u)
    u_prev = None

    # Iteratively update the eigenvector.
    while u_prev is None or np.linalg.norm(u - u_prev) > .000001:
        # Update u
        u_prev = u
        u = fast_dot(A, u)
        u_norm = np.linalg.norm(u)

        # Make u canonical length and orientation.
        u = u / u_norm
        if u[0] < 0:
            u = u * -1

    return (u, u_norm)

# Deflate a matrix by an eigenvector and eigenvalue.
def deflate(A, eigenvector, eigenvalue):
    ev = eigenvector[np.newaxis].T
    B = eigenvalue * fast_dot(ev, ev.T)
    return A - B
