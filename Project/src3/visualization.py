import numpy as np
import scipy
import sklearn.decomposition
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print('loading')

dat = np.load('../data2/data2.npz')
y = dat['Y_train']

print('PCA')

dat = np.load('pca.npz')
x_pca = dat['x']

plt.figure(figsize=(10, 5))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = y)
plt.savefig('pca.png')

print('t-SNE')

dat = np.load('tsne.npz')
x_tsne = dat['x']

plt.figure(figsize=(10, 5))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = y)
plt.savefig('tsne.png')