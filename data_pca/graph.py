# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:55:31 2016

@author: jonat
"""

from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt(open('all_data.csv'),delimiter=',', skiprows=1)

d = {}
d[10] = data[data[:, 0] == 10]
d[25] = data[data[:, 0] == 25]
d[50] = data[data[:, 0] == 50]
d[75] = data[data[:, 0] == 75]
d[100] = data[data[:, 0] == 100]

# Plot the accuracy in series by the number of components.
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(4, 1.5))
plt.title('accuracy of pca', fontsize=16)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('scale %', fontsize=16)
i = 0   
for k, w in d.items():
    v = w[w[:, 2] == 0]
    v = v[v[:,3] == 0]
    xs =(v[:, 1]*100).ravel()
    ys = ((v[:,6] + v[:,7]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='components=%d' % k)
    i += 1

plt.show()

# True positive
plt.figure(figsize=(4, 1.5))
plt.title('true positive rate of pca', fontsize=16)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('scale %', fontsize=16)
i = 0   
for k, v in d.items():
    v = v[v[:, 2] == 0]
    v = v[v[:,3] == 0]
    xs =(v[:, 1]*100).ravel()
    ys = ((v[:,6]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='components=%d' % k)
    i += 1

plt.show()

# True negative rate
plt.figure(figsize=(4, 1.5))
plt.title('true negative rate of pca', fontsize=16)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('scale %', fontsize=16)
i = 0   
for k, v in d.items():
    v = v[v[:, 2] == 0]
    v = v[v[:,3] == 0]
    xs =(v[:, 1]*100).ravel()
    ys = ((v[:,7]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='components=%d' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.15), ncol=1, fontsize=12)
plt.show()

plt.figure(figsize=(8, 1.5))
plt.title('effect of whitening on accuracy of pca', fontsize=12)
plt.ylabel('change in accuracy', fontsize=12)
plt.xlabel('scale %', fontsize=12)
i = 0   
for k, v in d.items():
    v = v[v[:, 2] == 0]
    l = v[v[:,3] == 1]
    r = v[v[:,3] == 0]
    xs =(l[:, 1]*100).ravel()
    ys = ((l[:,6] + l[:,7]) / 6000 * 100 - (r[:,6] + r[:,7]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='components=%d' % k)
    i += 1

plt.legend(loc='upper left', bbox_to_anchor=(0, -.4), ncol=3, fontsize=10)
plt.show()

plt.figure(figsize=(8, 1.5))
plt.title('running time of pca', fontsize=12)
plt.ylabel('seconds', fontsize=12)
plt.xlabel('scale %', fontsize=12)
i = 0   
for k, v in d.items():
    v = v[v[:, 2] == 0]
    l = v[v[:,3] == 0]
    xs =(l[:, 1]*100).ravel()
    ys = (l[:,5]).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='components=%d' % k)
    i += 1

plt.legend(loc='upper left', bbox_to_anchor=(0, -.4), ncol=3, fontsize=10)
plt.show()


