from sklearn.cluster import KMeans
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

print('k-means')

dat = np.load('tsne.npz')
x_tsne = dat['x']

kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE)
kmeans.fit(x_tsne)

labels = kmeans.labels_
true_labels = y.astype(int) - 1

dist = np.zeros((10, 10))
perm = list(range(10))
for i in range(10):
    x_i = x_tsne[y == i + 1, :]
    temp = kmeans.transform(x_i)
    dist[i, :] = np.mean(temp, 0)
    perm[i] = np.argmin(dist[i, :])


accuracy = float(np.sum([perm[j] for j in true_labels] == labels)) / float(np.sum(labels == labels))

print(accuracy)

plt.figure(figsize=(10, 5))
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c = labels)
plt.savefig('kmeans.png')