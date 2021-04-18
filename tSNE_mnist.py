#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 08:11:46 2020

@author: sjagadee
"""

import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import manifold
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

mnist_data = sklearn.datasets.load_digits()
X_train, X_test, y_train, y_test = model_selection.train_test_split(mnist_data.data, 
                                                    mnist_data.target, test_size=0.33)

tSNE = manifold.TSNE(perplexity=40, n_components=3)
Z_train = tSNE.fit_transform(X_train)
Z_main_dims = Z_train[:,0:3]

#Create 2D plot if number of components is 2
# plt.scatter(Z_main_dims[:,0], Z_main_dims[:,1], c = y_train, alpha=0.5)
# plt.show()

#Create 3D plot if we want to visualize 3 components
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = plt.Axes(projection='3d', fig=fig, rect=(0,0,12,12))
ax.scatter3D(Z_main_dims[:,0], Z_main_dims[:,1], Z_main_dims[:,2],
             s=150, c=y_train, alpha=0.5)
plt.show()