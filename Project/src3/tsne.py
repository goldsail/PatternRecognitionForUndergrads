import numpy as np
import scipy
import sklearn.decomposition
from sklearn.manifold import TSNE
import os

RANDOM_STATE = 2018

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

print('preprocessing using PCA')

x = sklearn.decomposition.PCA(n_components=10).fit_transform(x)

print('t-SNE')

x_tsne = TSNE(n_components=2, random_state=RANDOM_STATE).fit_transform(x)
np.savez('tsne.npz', x=x_tsne)
