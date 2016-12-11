import load_lfw
import neural_pair
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

print('hidden_nodes,output_nodes,resize,color,iterations,batch_size,choices,time,true_pos,true_neg,false_pos,false_neg,total')
for output_size in [25]:
	for resize, color in [(.05, False), (.1, False), (.15, False), (.2, False), (.25, False), (.3, False), (.05, True), (.1, True), (.15, True), (.2, True)]:
			for iterations, batch_size in [(20000, 1)]:#, (10000, 2), (5000, 4), (2500, 8), (1250, 16), (500, 40)]:
				for choices in [1]:
					train, test = neural_pair.generate_model(10, int(resize*120), int(resize*240), iterations=iterations, batch_size=batch_size, color=color, choices=choices, randomized_pairs=True)
					filename = 'resize%d_color%d.npy' % (int(100 * resize), int(color))

					f = partial(load_lfw.run_test, 1, train, test, 'funneled', resize, color=color, file=filename, crop=(120,240), mirror=False, ids=True)
					sys.stderr.write('output_size:%d, resize:%.2f, color:%d, iterations:%d, batch_size:%d, choices:%d\n' % (output_size, resize, int(color), iterations, batch_size, choices))
					#results = timeout(360, f)
					results=f()
					true_pos = results['true_pos']
					true_neg = results['true_neg']
					false_pos = results['false_pos']
					false_neg = results['false_neg']
					total = results['total']
					t1 = results['time']
					print('%d,%d,%.2f,%d,%d,%d,%d,%.3f,%d,%d,%d,%d,%d' % (70, output_size, resize, int(color), iterations, batch_size, choices, t1, true_pos, true_neg, false_pos, false_neg, total))
