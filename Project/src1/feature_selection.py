import numpy as np
import scipy
import sklearn.decomposition
import os

dat = np.load('../data1/data1.npz')
X_test = dat['X_test']
X_train = dat['X_train']
Y_train = dat['Y_train']
dat = []

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
x = np.divide(np.subtract(X_train, mean), std)

# method 1: PCA 0.99

pca = sklearn.decomposition.PCA(n_components=0.99)
X_train_pca_99 = pca.fit_transform(x)
X_test_pca_99 = pca.transform(np.divide(np.subtract(X_test, mean), std))

# method 2: Fisher 2000

P = np.size(x, axis=1)
N = np.size(x, axis=0)
Fisher = np.zeros((1, P))
for p in range(P):
    x1 = x[Y_train == 1, p]
    x2 = x[Y_train == 2, p]
    Fisher[0, p] = np.square(np.mean(x1) - np.mean(x2)) / (np.var(x1) + np.var(x2))

ind = np.argsort(-Fisher)

X_train_fisher_2000 = X_train[:, ind[0, range(2000)]]
X_test_fisher_2000 = X_test[:, ind[0, range(2000)]]

# save results

np.savez('pca_99.npz', X_train=X_train_pca_99, X_test=X_test_pca_99)
np.savez('fisher_2000.npz', X_train=X_train_fisher_2000, X_test=X_test_fisher_2000)
