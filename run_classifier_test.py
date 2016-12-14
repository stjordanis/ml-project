import load_lfw
import classifier
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
# .07, .12, .17, .22
# .1 -> 12, 24
# 
print('hidden_nodes,resize,color,iterations,batch_size,time,true_pos,true_neg,false_pos,false_neg,total,train_true_pos,train_true_neg,train_false_pos,train_false_neg')
for r in range(0, 1):	
	for resize, color in [ (0.12, False), (0.12, True)]:
		for iterations, batch_size in [(20000, 5), (20000, 10)]:
			print "new test"
			if resize == 0.07:
				width = 8
				height = 16
			elif resize == 0.12:
				width = 14
				height = 28
			elif resize == 0.17:
				width = 20
				height = 40
			else:
				width = 26
				height = 52
			train, test = classifier.generate_model(width, height, iterations, batch_size=batch_size, color=color)
			filename = 'resize%d_color%d.npy' % (int(100 * resize), int(color))

			f = partial(load_lfw.run_test, 1, train, test, 'funneled', resize, color=color, file=filename, crop=(120,240), mirror=False, ids=False)
			sys.stderr.write('resize:%.2f, color:%d, iterations:%d, batch_size:%d' % (resize, int(color), iterations, batch_size))
			#results = timeout(360, f)
			results=f()
			true_pos = results['true_pos']
			true_neg = results['true_neg']
			false_pos = results['false_pos']
			false_neg = results['false_neg']
			total = results['total']
			t1 = results['time']
			print('%d,%.2f,%d,%d,%d,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d' % (30, resize, int(color), iterations, batch_size, t1, true_pos, true_neg, false_pos, false_neg, total, results['tr_true_pos'], results['tr_true_neg'], results['tr_false_pos'], results['tr_false_neg']))
