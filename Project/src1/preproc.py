import numpy as np
import scipy
import os

X_train = np.loadtxt('../data1/2ctrainX.txt')
Y_train = np.loadtxt('../data1/2ctrainY.txt')
X_test = np.loadtxt('../data1/2ctestX.txt')

P = np.size(X_train, 1)
print(P)

ind = []
for p in range(P):
    temp = X_train[:, p]
    if np.var(temp) > 1e-7:
        ind.append(p)

X_train = X_train[:, ind]
X_test = X_test[:, ind]
print(np.size(X_train, 1))

np.savez('../data1/data1.npz', X_test = X_test, X_train = X_train, Y_train = Y_train)

