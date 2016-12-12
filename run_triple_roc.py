import load_lfw
import neural_triple as neural
from functools import partial
import errno, os, signal


train, test = neural.generate_model(10, int(.17*120), int(.17*240), iterations=20000, batch_size=4, color=False, consider=2, roc=1)
out = load_lfw.run_test(10, train, test, 'funneled', .17, color=False, file='resize17_color0.npy', crop=(120,240), mirror=False, roc=True, ids=True)
for (a, b) in out['roc']:
    print('%f,%d' % (a, int(b)))
