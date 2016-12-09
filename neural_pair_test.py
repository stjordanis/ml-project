import load_lfw
import neural_pair

train, test = neural_pair.generate_model(10, 24, 50, iterations=10001, batch_size=1, color=False, choices=2, randomized_pairs=True)
print(load_lfw.run_test(1, train, test, 'funneled', .2, False, file='sets24x50.npy', crop=(120,250), ids=True))

# All below are randomized_pairs=True
# 10 out, 24x50, 20001 iterations, batch size 1, choices=2 --> 428/600; 40001 iterations --> 180+254/600
# Same as above with choices=3: 199+200/600
# Same as above with chioces=None: 211+164/600
# Same as above, choices=3, 40001 iterations: 250+167/600
# At 5 choices, 400001 iterations, it becomes too hard. Can't get past 50/50.
# At 4 choices, 400001 iterations, 218+192/600

# 10 out, 36x75, 40001 iterations, choices=0, batch_size=1 --> 178+228/600

# 10 out, 2 choices, 1 batch, 20001 iterations, 24x50, randomized_pairs --> 175+236
# 


# Things to vary:
# Resolution
# Crop
# Number of iterations
# Size of output vector
# Batch size?
# Color vs. no color
# Choices
# Randomized vs. LFW pairs
# probability of picking same vs. different
# size of hidden layer
# perhaps even size of pooling and such
