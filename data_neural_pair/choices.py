# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:55:31 2016

@author: jonat
"""

from matplotlib import pyplot as plt
import numpy as np
import sys

data = np.loadtxt(open('output10_choices.csv'),delimiter=',', skiprows=1)


# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('candidates', fontsize=15)
i = 0   
xs =(data[:, 6]).ravel()
ys1 = data[:,17].ravel()*100
ys2 = data[:,20].ravel()*100
plt.plot(xs, ys1, '%s-o' % cs[1], label='validation' )
plt.plot(xs, ys2, '%s-o' % cs[2], label='training')
plt.show()

plt.figure(figsize=(2, 1.5))
plt.title('true pos', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('candidates', fontsize=15)
i = 0   
xs =(data[:, 6]).ravel()
ys1 = data[:,18].ravel()*100
ys2 = data[:,21].ravel()*100
plt.plot(xs, ys1, '%s-o' % cs[1], label='validation' )
plt.plot(xs, ys2, '%s-o' % cs[2], label='training')
plt.show()

plt.figure(figsize=(2, 1.5))
plt.title('true neg', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=5)
plt.xlabel('candidates', fontsize=15)
i = 0   
xs =(data[:, 6]).ravel()
ys1 = data[:,19].ravel()*100
ys2 = data[:,22].ravel()*100
plt.plot(xs, ys1, '%s-o' % cs[1], label='validation' )
plt.plot(xs, ys2, '%s-o' % cs[2], label='training')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()