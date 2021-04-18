#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:14:52 2020

@author: sjagadee
"""

import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import decomposition
import matplotlib.pyplot as plt

mnist_data = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = model_selection.train_test_split(mnist_data.data, 
                                                    mnist_data.target, test_size=0.33)

pca = decomposition.PCA()
Z_train = pca.fit_transform(X_train)
Z_main_dims = Z_train[:,0:2]
plt.scatter(Z_main_dims[:,0], Z_main_dims[:,1], c = y_train, alpha=0.5)
plt.show()

plt.plot(pca.explained_variance_ratio_)
plt.show()