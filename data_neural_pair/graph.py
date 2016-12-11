# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:55:31 2016

@author: jonat
"""

from matplotlib import pyplot as plt
import numpy as np
import sys

data = np.loadtxt(open('all_average.csv'),delimiter=',', skiprows=1)

d = {}
d['10'] = data[data[:, 0] == 10]
d['25'] = data[data[:, 0] == 25]


# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    xs =(v[:, 1]).ravel()
    ys = v[:,6].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (validation)' % k)
    ys = v[:,9].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (training)' % k)
    i += 2
plt.show()

# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true pos', fontsize=15)
#plt.ylabel('accuracy', fontsize=15)
plt.xlabel('scale %', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    xs =(v[:, 1]).ravel()
    ys = v[:,4].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (validation)' % k)
    ys = v[:,7].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (training)' % k)
    i += 2
plt.show()

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true neg', fontsize=15)
plt.xlabel('scale %', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    xs =(v[:, 1]).ravel()
    ys = v[:,5].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (valid.)' % k)
    ys = v[:,8].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (train.)' % k)
    i += 2
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy (color)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    xs =(v[:, 1]).ravel()
    ys = v[:,6].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (valid.)' % k)
    ys = v[:,9].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (train.)' % k)
    i += 2
plt.show()

# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true pos (color)', fontsize=15)
#plt.ylabel('accuracy', fontsize=15)
plt.xlabel('scale %', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    xs =(v[:, 1]).ravel()
    ys = v[:,4].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (validation)' % k)
    ys = v[:,7].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (training)' % k)
    i += 2
plt.show()

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true neg (color)', fontsize=15)
plt.xlabel('scale %', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    xs =(v[:, 1]).ravel()
    ys = v[:,5].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i], label='units=%s (valid.)' % k)
    ys = v[:,8].ravel()*100
    plt.plot(xs, ys, '%s-o' % cs[i+1], label='units=%s (train.)' % k)
    i += 2
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()