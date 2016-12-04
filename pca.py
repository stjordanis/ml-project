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
	# Zheng paper is broken.
	def __init__(self, X, y, num_components, save=False, zheng_paper=False):
		if zheng_paper:
			self.zheng_algorithm(X, y, num_components, save)
			return

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

		perror('classes: %d' % len(classes.keys()))
		perror('n: %d' % n)
		perror('features: %d' % d)

		# Calculate a centered matrix.
		Z = np.zeros(X.shape)
		for i in range(n):
			Z[i] = X[i] - means[y[i]]
		Z = Z / d

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
		# Problem: When the number of features exceeds the number of samples,
		# Sw will be singular (non-invertible).
		# There are many solutions suggested by the literature:
		# 1) Fisherfaces says to use PCA to reduce the feature space down to
		#    size N-c. Doing so is prohibitively expensive for the number of
		#    features we're using.
		# 2) Others suggest using the inverse approximation. This doesn't seem
		#    to yield useful results.
		# 3) We can work on very low-resolution images. Seems to work reasonably
                #    well at size 25x25. Cropping helps (try (280, 250)). It seems like it
                #    generally works better on lower resolution. 
                #    Initial best results: 10 components at .1 resize at (115, 250) crop.
		perror('Determinant of Sw: %g' % np.linalg.det(Sw))
		t0 = time()
		A = fast_dot(np.linalg.pinv(Sw), Sb)
		perror('Inverted Sw and multiplied it by Sb in %.3f seconds.' % (time() - t0))

		# Find the top eigenvectors.
		i = 0
		while len(components) < num_components:
			eigenvector, eigenvalue = next_eigenvector(A, threshold=.00001)
			components.append(eigenvector)
			A = deflate(A, eigenvector, eigenvalue)
			if True:
				i += 1
				scipy.misc.imsave(str(i) + 'fisher.jpg', eigenvector.reshape([11, 25]))

		self.components = np.array(components)


	def zheng_algorithm(self, X, y, k, save):
		# Basic initialization.
		n, d = X.shape
		components = []

		# Calculate the mean face.
		self.mu = np.mean(X, axis=0)

		# Count the number of classes
		idx = 0
		c2i = {}
		for yi in y:
			if yi not in c2i:
				c2i[yi] = idx
				idx += 1
		c = len(c2i.keys())
		print('Classes: %d' % c)

		# Calculate the counts and means for each class.
		pi = np.zeros((c, 1)) # Column vector of class counts.
		E = np.zeros((n, c)) # Indicator of whether input i is in class j.
		for i, yi in enumerate(y):
			pi[c2i[yi]][0] += 1
			E[i][c2i[yi]] = 1

		# Matrices derived from pi and friends.
		pi_sqrt = np.sqrt(pi)
		PI = np.diag(pi.ravel())
		PI_sqrt = np.diag(pi_sqrt.ravel())
		H_pi = np.eye(c) - pi.dot(pi.T) / n
		H_n = np.eye(n) - 1/n
		M = fast_dot(fast_dot(np.linalg.inv(PI), E.T), X)

		# Method cited as "previous way of doing things"
		t0 = time()
		St = fast_dot(fast_dot(X.T, H_n), X)
		Sb = fast_dot(fast_dot(fast_dot(fast_dot(fast_dot(fast_dot(X.T, H_n), E), np.linalg.inv(PI)), E.T), H_n), X)
		A = fast_dot(np.linalg.pinv(St), Sb)

		#from IPython.core.debugger import Tracer

		#t0 = time()
		#sig2 = 0
		#G = fast_dot(fast_dot(fast_dot(H_n, X), X.T), H_n) + sig2 * np.eye(n)
		#G = fast_dot(fast_dot(fast_dot(fast_dot(X.T, H_n), np.linalg.inv(G)), E), PI_sqrt)
		#A = fast_dot(fast_dot(fast_dot(fast_dot(G, PI_sqrt), E.T), H_n), X)
		#perror('Calculated A in %.3f seconds.' % (time() - t0))
		#print(A.shape)
		#Tracer()()

		# Find the top eigenvectors.
		i = 0
		while len(components) < k:
			eigenvector, eigenvalue = next_eigenvector(A, threshold=.00001)
			components.append(eigenvector)
			A = deflate(A, eigenvector, eigenvalue)
			if False:
				i += 1
				scipy.misc.imsave(str(i) + 'fisher.jpg', eigenvector.reshape([75, 75]))

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
def next_eigenvector(A, threshold=.000001):
    n, d = A.shape

    # Initialze the eigenvector randomly.
    u = np.random.random([d])
    u = u / np.linalg.norm(u)
    u_prev = None

    # Iteratively update the eigenvector.
    while u_prev is None or np.linalg.norm(u - u_prev) > threshold:
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
