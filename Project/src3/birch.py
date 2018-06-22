from sklearn.cluster import Birch
import numpy as np
import scipy
import sklearn.decomposition
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import random

RANDOM_STATE = 2018

print('loading')

dat = np.load('../data2/data2.npz')
y = dat['Y_train']

print('birch')

dat = np.load('tsne.npz')
x_tsne = dat['x']

birch = Birch(n_clusters=10, threshold=0.4, branching_factor=50)
birch.fit(x_tsne)

labels = birch.labels_
true_labels = y.astype(int) - 1

dist = np.zeros((10, 10))
perm = list(range(10))
for i in range(10):
    x_i = x_tsne[y == i + 1, :]
    temp = birch.predict(x_i)
    perm[i] = scipy.stats.mode(temp, axis=None)[0][0]


accuracy = float(np.sum([perm[j] for j in true_labels] == labels)) / float(np.sum(labels == labels))

print(accuracy)

plt.figure(figsize=(10, 5))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = labels)
plt.savefig('birch.png')