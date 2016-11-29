#######################################################################
# EIGENFACES                                                          #
# ------------------------------------------------------------------- #
# Performs facial recognition on the LFW dataset using PCA and the    #
# eigenfaces algorithm.                                               #
#######################################################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression as lr

from pca import pca, transform
from time import time

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

out = run_test(150, 'logistic')
print(classification_report(t2, out, target_names=t_names))
print('Accuracy: %f' % sklearn.metrics.accuracy_score(t2, out))