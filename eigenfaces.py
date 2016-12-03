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
	def train(set, classes):
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
		Y = transform(faces, components, faces)
		perror('Transformed all faces into the component space.')

		# Train a classifier.
		if classifier == 'logistic':
			t0 = time()
			perror('Performing logistic regression.')
			model = lr(solver='lbfgs', multi_class='multinomial')
			model.fit(Y, names)
			perror('Logistic regression completed in %.3f seconds.' % (time() - t0))
		elif classifier == 'knn':
			
		

		return model, components, faces

	def same_or_different(face1, face2, m):
		model, components, faces = m
		f1 = np.ravel(face1)
		f2 = np.ravel(face2)
		y1 = model.predict(transform(f1.reshape(1, -1), components, faces))[0]
		y2 = model.predict(transform(f2.reshape(1, -1), components, faces))[0]
		return y1 == y2

	return train, same_or_different

#train, test = create_train_test(50)
#print(load_lfw.run_test_unrestricted(1, train, test, type='funneled', resize=.3, color=False))

'''
# Load the LFW images.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=.4)

# Load the faces.
n, h, w = lfw_people.images.shape
X = lfw_people.data.astype('float64')
d = X.shape[1]

# Load the targets.
t = lfw_people.target
t_names = lfw_people.target_names
classes = t_names.shape[0]

# Print basics.
print('Dataset Size')
print('-'*20)
print('faces\t\t%d' % n)
print('pixels\t\t%d'% d)
print('(w, h)\t\t(%d, %d)' % (w, h))
print('identities\t%d' % int(classes))

# Split into train and test sets.
X1, X2, t1, t2 = train_test_split(X, t, test_size=.25, random_state=0)

# Function for running a test.
def run_test(k, classifier):
    components = np.array(pca(X1, k))
    #comps = sklearn.decomposition.PCA(n_components=k, svd_solver='randomized').fit(X1)
    #eigenfaces = components.reshape((k, h, w))
    
    Y1 = transform(X1, components, X1)
    Y2 = transform(X2, components, X1)
    
    #print(comps.components_ - components)
    
    
    #Y1 = comps.transform(X1)
    #Y2 = comps.transform(X2)
    
    if classifier == 'logistic':
        model = lr(multi_class='multinomial', solver='lbfgs')
        model.fit(Y1, t1)
        y2 = model.predict(Y2)
    elif classifier == 'svm':
        svm = sklearn.svm.SVC()
        svm.fit(Y1, t1)
        y2 = svm.predict(Y2)
    elif classifier == 'original':
        # Prepare for the original classifier.
        per_class_count = np.zeros(classes)
        per_class_proj = np.zeros((classes, k))
        
        for i in range(Y1.shape[0]):
            per_class_proj[t1[i]] = (per_class_proj[t1[i]] * per_class_count[t1[i]] + Y1[i]) / (per_class_count[t1[i]] + 1)
            per_class_count[t1[i]] += 1
        
        y2 = np.argmin(pairwise_distances(Y2, per_class_proj, metric='euclidean'), axis=1)
    
    return y2

t0 = time()
out = run_test(150, 'logistic')
print(time() - t0)
print(classification_report(t2, out, target_names=t_names))
print('Accuracy: %f' % sklearn.metrics.accuracy_score(t2, out))
'''
