import load_lfw
import eigenfaces
from functools import partial
import errno, os, signal


train, test = eigenfaces.instance(75, classifier='logistic', feature='distance', dim_reduction='pca', whiten=False, roc=True)
out = load_lfw.run_test(10, train, test, 'funneled', .2, color=False, file='resize20_color0.npy', crop=(120,250), mirror=False, roc=True)
for (a, b) in out['roc']:
    print('%f,%d' % (a, int(b)))
