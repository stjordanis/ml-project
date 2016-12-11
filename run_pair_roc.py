import load_lfw
import neural_pair2 as neural_pair
from functools import partial
import errno, os, signal


train, test = neural_pair.generate_model(10, int(.12*120), int(.12*240), iterations=2500, batch_size=8, color=True, consider=1, roc=1)
out = load_lfw.run_test(10, train, test, 'funneled', .12, color=True, file='resize12_color1.npy', crop=(120,240), mirror=False, roc=True, ids=True)
for (a, b) in out['roc']:
    print('%f,%d' % (a, int(b)))
