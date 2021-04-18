#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:14:52 2020

@author: sjagadee
"""

import numpy as np
import sklearn
from sklearn import model_selection
import matplotlib.pyplot as plt

mnist_data = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = model_selection.train_test_split(mnist_data.data, 
                                                    mnist_data.target, test_size=0.33)


#Compute pca by hand instead of using scikit learn
cov_x = np.cov(X_train, rowvar=False) #column is variable 
lambdas, V = np.linalg.eig(cov_x)

idx = np.argsort(-lambdas) #negating the matrix to sort in descending order
lambdas_sorted = lambdas[idx]
lambdas_sorted = np.maximum(lambdas_sorted, 0) #get rid of very small negative vals
V_sorted = V[:,idx]

Q = V_sorted # We know this is Q

Z_train_manual = X_train.dot(Q)
Z_main_dims_manual = Z_train_manual[:,0:2]
plt.scatter(Z_main_dims_manual[:,0], Z_main_dims_manual[:,1], c = y_train, alpha=0.5)
plt.show()

# plot variances
plt.plot(lambdas_sorted)
plt.title("Variance of each component")
plt.show()