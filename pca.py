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

# Assumes that the feature vectors x(i) are in rows, and each
# column represents a specific feature. There are n feature
# vectors and d features in each vector.
#
# Returns a 2D array containing the principal components (a basis).
# The components array is (k x d), where (k = num_components) and d
# is the number of features.
def pca(X, num_components):
    components = []
    n, d = X.shape
    
    # Create the covariance matrix.
    mu = np.mean(X, axis=0)
    Z = X - mu
    A = fast_dot(Z.T, Z) / d

    while len(components) < num_components:
        eigenvector, eigenvalue = next_eigenvector(A)
        components.append(eigenvector)
        A = deflate(A, eigenvector, eigenvalue)
    
    return np.array(components)



# Projects the matrix X (same as in pca) onto the coordinate system
# specified by components.
def transform(X, components, Xtr):
    mu = np.mean(Xtr, axis=0)
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
