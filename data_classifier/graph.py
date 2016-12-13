from matplotlib import pyplot as plt
import numpy as np
import sys

data = np.loadtxt(open('classifier_results.csv'),delimiter=',', skiprows=1)

d = {}
d[10] = data[data[:, 10] == 10]
d[3] = data[data[:, 10] == 3]
d[5] = data[data[:, 10] == 5]
d[20] = data[data[:, 10] == 20]
d[40] = data[data[:, 10] == 40]

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=15)
plt.tick_params(labelsize=14)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=15)
i = 0   
for k, w in d.items():
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16] + v[:,17]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16] + v[:,17]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 0] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
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
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16] + v[:,17]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
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
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
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
    v = w[w[:, 12] == 0] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16] + v[:,17]) / 6000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,16]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigenvecs=%d' % k)
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
    v = w[w[:, 12] == 1] # No color
    v = v[v[:, 14] == 1] # No mirror
    xs =(v[:, 11]*100).ravel()
    ys = ((v[:,17]) / 3000 * 100).ravel()
    plt.plot(xs, ys, '%s-o' % cs[i], label='eigvecs=%d' % k)
    i += 1
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1), ncol=1, fontsize=12)
plt.show()