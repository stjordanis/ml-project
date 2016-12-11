# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:55:31 2016

@author: jonat
"""

from matplotlib import pyplot as plt
import numpy as np
import sys

data = np.loadtxt(open('all.csv'),delimiter=',', skiprows=1)

d = {}
d['5, 50'] = data[data[:, 10] == 5]
d['5, 100'] = data[data[:, 10] == 5]
d['10, 50'] = data[data[:, 10] == 10]
d['10, 100'] = data[data[:, 10] == 10]

d['5, 50'] = d['5, 50'][d['5, 50'][:, 11] == 50]
d['5, 100'] = d['5, 100'][d['5, 100'][:, 11] == 100]
d['10, 50'] = d['10, 50'][d['10, 50'][:, 11] == 50]
d['10, 100'] = d['10, 100'][d['10, 100'][:, 11] == 100]


# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17] + v[:,18]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='fda,pca=%s' % k)
    i += 1

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
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

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
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,18]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='pca,fda=%s' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()




###############################################################################
# SAME AS ABOVE BUT WITH COLOR                                                #
###############################################################################
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy (color)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17] + v[:,18]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

plt.show()


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true pos (color)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

plt.show()

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true neg (color)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 0] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,18]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='lda,pca=%s' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()


############################
###########################
############################
# Plot the accuracy in series by the number of components. No color, mirroring
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy (mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17] + v[:,18]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

plt.show()


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true pos (mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

plt.show()

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true neg (mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 0] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,18]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='fda,pca=%s' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()


###############################################################################
# SAME AS ABOVE BUT WITH COLOR                                                #
###############################################################################
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy (color+mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17] + v[:,18]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i])
    i += 1

plt.show()


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true pos (color+mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='pca,fda=%s' % k)
    i += 1

plt.show()

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('true neg (color+mirror)', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=3)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 13] == 1] # No color
    v = v[v[:, 15] == 1] # No mirror
    xs =(v[:, 12]*100).ravel()
    ys = ((v[:,18]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='pca,fda=%s' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()