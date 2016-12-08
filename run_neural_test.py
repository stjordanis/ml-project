import load_lfw
import shared

train, test = shared.generate_model(10, 24, 50, iterations=10000, batch_size=1)
print(load_lfw.run_test(1, train, test, 'funneled', .2, False, file='sets24x50.npy', crop=(120,250)))
