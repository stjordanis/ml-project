import load_lfw
import neural_triple as neural

train, test = neural.generate_model(20, 48, 100, iterations=4001, batch_size=3, color=True, consider=2)
print(load_lfw.run_test(1, train, test, 'funneled', .4, True, file='sets48x100.npy', crop=(120,250), ids=True))

# 253 + 179
#  10?, 24x50, 20001 iterations, batch_size 1, color False, consider 2, 5000 for lr

# 254 + 194
#  10?, 36x75, 5001 iterations, batch_size 5, color False, consider 2, 5000 for lr

# 244 + 183
# 10?, 24x50, 15001 iterations, batch_size 1, color True, consider 2, 5000 for lr
# 178 + 220
# 7001 iterations

# 235+182
# 20, 48x100, 4001 iterations, batch_size 3, color=True, consider=2, crop=120x250
