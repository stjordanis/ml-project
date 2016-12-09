import load_lfw
import eigenfaces

train, test = eigenfaces.instance(5, dim_reduction='fisher_pca_100')
print(load_lfw.run_test(10, train, test, 'funneled', .3, color=False, file='sets36x75.npy', crop=(120,250), mirror=False))

