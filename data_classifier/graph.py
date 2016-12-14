from matplotlib import pyplot as plt
import numpy as np
import sys

#data = np.loadtxt(open('dc.csv'),delimiter=',')

# average the three trials
data2 = np.loadtxt(open('trials.csv'),delimiter=',')
avg = []
for i in range(0, 32):
    avg.append(np.mean(data2[3*i : 3*i + 3], axis=0))
#print(np.asarray(avg))

data = np.asarray(avg)

d = {}
# iterations
d[10000] = data[data[:, 3] == 10000]
d[20000] = data[data[:, 3] == 20000]

#print(d[12])
#d[7] = data[data[:, 1] == 0.07]
#d[22] = data[data[:, 1] == 0.22]
#d[17] = data[data[:, 1] == 0.17]
'''
# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 5, NO COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6] + v[:,7]) / 600 * 100).ravel() #validation accuracy
    ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('acc-b5c0.png')

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('accuracy', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 10, NO COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6] + v[:,7]) / 600 * 100).ravel() #validation accuracy
    ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('acc-b10c0.png')


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true pos', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 5, NO COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,11] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tp-b5c0.png')


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true pos', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE POSITIVE, BATCH SIZE = 10, NO COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,11] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tp-b10c0.png')
'''

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true neg', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE NEGATIVE, BATCH SIZE = 5, NO COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,7]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,12] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
#plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
plt.show()
fig.savefig('tn-b5c0n.png')

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true neg', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE NEGATIVE, BATCH SIZE = 10, NO COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,7]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,12] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='validation, iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='training, iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
#plt.legend( ncol=1, fontsize=12)
plt.show()
fig.savefig('tn-b10c0n.png')
'''
###############################################################################
# SAME AS ABOVE BUT WITH COLOR                                                #
###############################################################################

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('accuracy (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 5, COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6] + v[:,7]) / 600 * 100).ravel() #validation accuracy
    ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('acc-b5c1_.png')

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('accuracy (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 10, COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 1] # No color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6] + v[:,7]) / 600 * 100).ravel() #validation accuracy
    ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('acc-b10c1_.png')


# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true pos (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE POSITIVE, BATCH SIZE = 5, COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,11] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tp-b5c1_.png')

# True positive
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true pos (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE POSITIVE, BATCH SIZE = 10, COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,6]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,11] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tp-b10c1_.png')
'''
# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true neg (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE NEGATIVE, BATCH SIZE = 5, COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,7]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,12] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
#plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
plt.show()
fig.savefig('tn-b5c1_n.png')

# True negative rate
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('true neg (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("TRUE NEGATIVE, BATCH SIZE = 10, COLOR, ")
for k, w in d.items():
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = ((v[:,7]) / 300 * 100).ravel() #accuracy
    ys2 = ((v[:,12] ) / 2700 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[2*i], label='iterations=%d' % k)
    print("validation, iterations = " , k, "color = ", cs[2*i])
    plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
    print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
#plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.25), ncol=1, fontsize=12)
plt.show()
fig.savefig('tn-b10c1_n.png')
'''
##### TIME TAKEN

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('time taken', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 5, NO COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 0] # No color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = (v[:,5]).ravel() #validation accuracy
   # ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[i], label='iterations=%d' % k)
    print("iterations = " , k, "color = ", cs[i])
   # plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
   # print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tt-b5c0.png')


# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('time taken', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 10, NO COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 0] # color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = (v[:,5]).ravel() #validation accuracy
   # ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[i], label='iterations=%d' % k)
    print("iterations = " , k, "color = ", cs[i])
   # plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
   # print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tt-b10c0.png')



# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('time taken (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 5, COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 1] # No color
    v = v[v[:, 4] == 5] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = (v[:,5]).ravel() #validation accuracy
   # ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[i], label='iterations=%d' % k)
    print("iterations = " , k, "color = ", cs[i])
   # plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
   # print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.show()
fig.savefig('tt-b5c1_.png')
'''

# Plot the accuracy in series by the number of components. No color
cs = ['r', 'g', 'b', 'k', 'c', 'm'] 
fig = plt.figure(figsize=(2, 1.5))
plt.title('time taken (color)', fontsize=6)
plt.tick_params(labelsize=6)
plt.locator_params(nbins=4)
plt.xlabel('scale %', fontsize=6)
plt.tight_layout()
i = 0   
print("BATCH SIZE = 10, COLOR, ")
for k, w in d.items():
    # separate out our data
    v = w[w[:, 2] == 1] # color
    v = v[v[:, 4] == 10] # batch size
    xs = (v[:, 1]*100).ravel() # scale
    ys = (v[:,5]).ravel() #validation accuracy
   # ys2 = ((v[:,11] + v[:,12]) / 5400 * 100).ravel() # training accuracy
    plt.plot(xs, ys, '%s-o' % cs[i], label='iterations=%d' % k)
    print("iterations = " , k, "color = ", cs[i])
   # plt.plot(xs, ys2, '%s-o' % cs[2*i+1], label='iterations=%d' % k)
   # print("training, iterations = " , k, "color = ", cs[2*i + 1])
    i += 1
plt.legend( ncol=1, fontsize=12)
plt.show()
#fig.savefig('tt-b10c1_.png')

