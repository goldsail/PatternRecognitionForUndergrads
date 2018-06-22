import numpy as np
import scipy
import sklearn.decomposition
import os
from sklearn import neural_network
from sklearn import cross_validation
import random

random.seed(2018)

dat = np.load('../data1/data1.npz')
X_test = dat['X_test']
X_train = dat['X_train']
Y_train = dat['Y_train']

# Fully connected NN

## fisher 2000

dat = np.load('fisher_2000.npz')
X_train_fisher_2000 = dat['X_train']
X_test_fisher_2000 = dat['X_test']

fc_nn = neural_network.MLPClassifier(hidden_layer_sizes=(1000), learning_rate='adaptive', learning_rate_init=0.01)

scores = cross_validation.cross_val_score(fc_nn, X_train_fisher_2000, Y_train, cv=5)
print(scores)

fc_nn.fit(X_train_fisher_2000, Y_train)
Y_pred = fc_nn.predict(X_test_fisher_2000)
np.savetxt('Y_pred_fc_nn_fisher_2000.txt', Y_pred, fmt='%d')

## pca 99

dat = np.load('pca_99.npz')
X_train_pca_99 = dat['X_train']
X_test_pca_99 = dat['X_test']

fc_nn = neural_network.MLPClassifier(hidden_layer_sizes=(1000), learning_rate='adaptive', learning_rate_init=0.01)

scores = cross_validation.cross_val_score(fc_nn, X_train_pca_99, Y_train, cv=5)
print(scores)

