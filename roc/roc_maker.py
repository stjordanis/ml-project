# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 03:45:03 2016

@author: jonat
"""

import numpy as np
import matplotlib.pyplot as plt

def positives(data, threshold):
    positives = data[:][data[:,1] > 0]
    tp_rate = positives[positives[:,0] >= threshold].shape[0] / positives.shape[0]
    
    negatives = data[:][data[:,1] == 0]
    fp_rate = negatives[negatives[:,0] >= threshold].shape[0] / negatives.shape[0]
    
    return tp_rate,fp_rate
    

def addroc(filename, label):
    data = np.loadtxt(open(filename),delimiter=',')
    
    # find the min and max distances.
    dmin = np.min(data, axis=0)[0]
    dmax = np.max(data, axis=0)[0]

    # Define an increment.
    n = 1000
    incr = (dmax - dmin) / 1000
    
    xs, ys = [], []
    
    for i in range(n):
        x, y = positives(data, dmin + i * incr)
        xs.append(x)
        ys.append(y)
        
    plt.plot(xs, ys, '-', label=label)

        
    

plt.figure(figsize=(6.8, 3))
plt.title('ROC for Facial Recognition', fontsize=15)
plt.tick_params(labelsize=14)
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)

addroc('pca.csv', 'PCA (75 components, 20% scaling)')
addroc('fisher.csv', 'FDA (5 eigenvectors, 7% scaling, color+mirroring)')
addroc('fisher_pca.csv', 'PCA+FDA (100 components, 5 eigenvectors, 20% scaling)')
addroc('pair_avg.csv', 'Pair CNN (10 output, 12% scaling, color, 8 per batch, 2.5k iterations)')

plt.legend(loc='upper left', bbox_to_anchor=(-.1, -.2), ncol=1, fontsize=11)

plt.show()