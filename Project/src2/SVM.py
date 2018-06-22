import numpy as np
import scipy
import sklearn.decomposition
import os
from sklearn import svm
from sklearn import cross_validation
import random

random.seed(2018)

dat = np.load('../data2/data2.npz')
X_test = dat['X_test']
X_train = dat['X_train']
Y_train = dat['Y_train']

# Linear SVM

## fisher 2000

dat = np.load('fisher_2000.npz')
X_train_fisher_2000 = dat['X_train']
X_test_fisher_2000 = dat['X_test']

linear_svm = svm.SVC(kernel='linear')

print('start training')

scores = cross_validation.cross_val_score(linear_svm, X_train_fisher_2000, Y_train, cv=5)
print(scores)

linear_svm.fit(X_train_fisher_2000, Y_train)
Y_pred = linear_svm.predict(X_test_fisher_2000)
np.savetxt('Y_pred_linear_svm_fisher_2000.txt', Y_pred, fmt='%d')

