import load_lfw
import eigenfaces
from functools import partial
import errno, os, signal

class TimeoutError(Exception):
    pass
import sys

def handler(signum, frame):
    raise TimeoutError('timeout')
def timeout(seconds, f):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        result = f()
    except:
        sys.stderr.write('Timed out.\n')
        result =  {'true_pos':0,'true_neg':0,'false_pos':0, 'false_neg':0, 'total':0,'time':-1}
    finally:
        signal.alarm(0)
    return result

print('det,fisher_components,pca_components,resize,color,whiten,mirror,time,true_pos,true_neg,false_pos,false_neg,total')
for (pc, fc) in [(200, 5)]:
    for resize, color in [(.07, True)]:
            for whiten in [False]:
                for mirror in [True]:
                    train, test = eigenfaces.instance(fc, classifier='logistic', feature='distance', dim_reduction='fisher_pca_%d' % pc, whiten=whiten)
                    filename = 'resize%d_color%d.npy' % (int(10 * resize), int(color))

                    f = partial(load_lfw.run_test, 10, train, test, 'funneled', resize, color=color, file=filename, crop=(120,250), mirror=mirror)
                    sys.stderr.write('fc,pc: %d,%d, resize: %.1f, color: %d, whiten: %d, mirror: %d\n' % (fc,pc, resize, int(color), int(whiten), int(mirror)))
                    results = timeout(360, f)
                    #results = f()
                    true_pos = results['true_pos']
                    true_neg = results['true_neg']
                    false_pos = results['false_pos']
                    false_neg = results['false_neg']
                    total = results['total']
                    t1 = results['time']
                    print('%d,%d,%.2f,%d,%d,%d,%.3f,%d,%d,%d,%d,%d' % (fc, pc, resize, int(color), int(whiten), int(mirror), t1, true_pos, true_neg, false_pos, false_neg, total))
