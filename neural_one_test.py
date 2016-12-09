import load_lfw
import neural_one

train, test = neural_one.generate_model(10, 24, 50, iterations=40001, batch_size=1, color=False, choices=None, randomized_pairs=True)
print(load_lfw.run_test(1, train, test, 'funneled', .2, False, file='sets24x50.npy', crop=(120,250), ids=True))

