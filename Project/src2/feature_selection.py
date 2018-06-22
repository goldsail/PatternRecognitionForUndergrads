import numpy as np
import scipy
import sklearn.decomposition
import os

print('loading')

dat = np.load('../data2/data2.npz')
X_test = dat['X_test']
X_train = dat['X_train']
Y_train = dat['Y_train']
dat = []

print('normalizing')

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
x = np.divide(np.subtract(X_train, mean), std)

print('Fisher')

# method 1: Fisher 2000

P = np.size(x, axis=1)
N = np.size(x, axis=0)
Fisher = np.zeros((1, P))

for p in range(P):
    Fisher[0, p] = 0
    if (p % 100 == 0):
        print(p)
    for i in range(1, 10):
        for j in range(i+1, 11):
            x1 = x[Y_train == i, p]
            x2 = x[Y_train == j, p]
            Fisher[0, p] = np.max([
                Fisher[0, p], 
                (np.square(np.mean(x1) - np.mean(x2)) + 0.01) / (np.var(x1) + np.var(x2) + 0.01)
            ])


ind = np.argsort(-Fisher)

X_train_fisher_2000 = X_train[:, ind[0, range(2000)]]
X_test_fisher_2000 = X_test[:, ind[0, range(2000)]]

# method 2: Fisher 4000

X_train_fisher_4000 = X_train[:, ind[0, range(4000)]]
X_test_fisher_4000 = X_test[:, ind[0, range(4000)]]


print('Saving')

# save results
np.savez('fisher_2000.npz', X_train=X_train_fisher_2000, X_test=X_test_fisher_2000)
np.savez('fisher_4000.npz', X_train=X_train_fisher_4000, X_test=X_test_fisher_4000)