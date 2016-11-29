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
    A = Z.T.dot(Z) / d
    
    while len(components) < num_components:
        eigenvector, eigenvalue = next_eigenvector(A)
        components.append(eigenvector * (1 / np.sqrt(eigenvalue)))
        A = deflate(A, eigenvector, eigenvalue)
    
    return np.array(components)

# Projects the matrix X (same as in pca) onto the coordinate system
# specified by components.
def transform(X, components, Xtr):
    mu = np.mean(Xtr, axis=0)
    Z = X - mu
    
    return Z.dot(components.T)
    
# Use the power method to calculate an eigenvector of X.
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
        u = A.dot(u)
        u_norm = np.linalg.norm(u)
        
        # Make u canonical length and orientation.
        u = u / u_norm
        if u[0] < 0:
            u = u * -1
    
    return (u, u_norm)

# Deflate a matrix by an eigenvector and eigenvalue.
def deflate(A, eigenvector, eigenvalue):
    ev = eigenvector[np.newaxis].T
    B = eigenvalue * ev.dot(ev.T)
    return A - B
    
    
# Testing code.
#from sklearn.decomposition import PCA as sklearn_pca
#A = np.array([[1, 2, 1], [6, -1, 0], [-1, -2, -1], [4, 5, 6], [8, 9, 10]])
#print(pca(A, 5))
#print('ALT IS BROKEN')
#print(pca(A, 2, alt=True))
#skp = sklearn_pca(n_components=2)
#skp.fit(A)
#print(skp.components_)