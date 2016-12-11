import load_lfw
import eigenfaces
from functools import partial
import errno, os, signal


train, test = eigenfaces.instance(5, classifier='logistic', feature='distance', dim_reduction='fisher', whiten=False, roc=True)
out = load_lfw.run_test(10, train, test, 'funneled', .07, color=True, file='resize7_color1.npy', crop=(120,250), mirror=True, roc=True)
for (a, b) in out['roc']:
    print('%f,%d' % (a, int(b)))
