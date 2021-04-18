#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:40:13 2020

@author: sjagadee

Recreated from original code by lazy programmer for practicing purposes
# https://www.udemy.com/unsupervised-deep-learning-in-python
"""

import tensorflow as tf
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn import datasets

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

def init_weights(l1, l2):
    weights = np.random.randn(l1, l2) * np.sqrt(1/l1)
    return weights.astype(np.float32)

def T_shared_zeros_like32(p):
    # p is a Theano shared itself
    return theano.shared(np.zeros_like(p.get_value(), dtype=np.float32))

def momentum_updates(cost, params, mu, learning_rate):
    # momentum changes
    dparams = [T_shared_zeros_like32(p) for p in params]

    updates = []
    grads = T.grad(cost, params)
    for p, dp, g in zip(params, dparams, grads):
        dp_update = (mu*dp - learning_rate*g).astype('float32')
        p_update = (p + dp_update).astype('float32')

        updates.append((dp, dp_update))
        updates.append((p, p_update))
    return updates
    

class AutoEncoder(object):
    def __init__(self, M, ann_id):
        #M is hidden units and ann_id is name/id of the network
        self.M = M
        self.ann_id = ann_id            
        
    def fit(self, X, lr = 0.1, momentum=0.99, epochs=1, batch_sz=100):
        momentum = np.float32(momentum)
        lr = np.float32(lr)
        
        #get the dimension of input data and number of batches
        N, D = X.shape
        n_batches = N // batch_sz
        
        #init the weights and biases
        W0 = init_weights(D, self.M)
        self.w = theano.shared(W0, 'w_' + self.ann_id)
        self.bh = theano.shared(np.zeros(self.M, dtype=np.float32), 'bh_' + self.ann_id)
        self.bo = theano.shared(np.zeros(D, dtype=np.float32), 'bo_' + self.ann_id)
        self.params = [self.w, self.bh, self.bo]        

        #keep track of changes in these vars to calculate momentum
        self.dw = theano.shared(np.zeros(W0.shape), 'w_' + self.ann_id)
        self.dbh = theano.shared(np.zeros(self.M), 'bh_' + self.ann_id)
        self.dbo = theano.shared(np.zeros(D), 'bo_' + self.ann_id)
        self.dparams = [self.dw, self.dbh, self.dbo]
        
        #Create symbolic variables for using it in cost and momentum calculations
        X_in = T.matrix('X_' + self.ann_id)
        Xhat = self.forward_output(X_in)
        
        #cross-entropy cost function
        cost = -(X_in * T.log(Xhat) + (1 - X_in) * T.log(1 - Xhat)).flatten().mean()
        cost_operation = theano.function(inputs=[X_in], outputs=cost,)    
        
        #momentum updates for SGD
        mom_updates = momentum_updates(cost, self.params, momentum, lr)
        train_operation = theano.function(inputs=[X_in], updates=mom_updates,)
        
        #train the auto encoder
        self.costs = []
        from sklearn.utils import shuffle
        for i in range(epochs):
            print("epoch {0} / {1}".format(i,epochs))
            X = shuffle(X)
            for j in range(n_batches):
                batch_data = X[j*batch_sz:(j*batch_sz + batch_sz)]
                train_operation(batch_data)
                cost_batch = cost_operation(batch_data)
                if j % 20 == 0:
                    print("epoch {0} batch {1} : cost {2}".format(i, j, cost_batch))
                self.costs.append(cost_batch)
                
        self.predict = theano.function(inputs=[X_in], outputs=Xhat)        
                
    def forward_hidden(self, X):
        Z = T.nnet.sigmoid(X.dot(self.w) + self.bh)
        return Z
    
    def forward_output(self,X):
        Z = self.forward_hidden(X)
        Y = T.nnet.sigmoid(Z.dot(self.w.T) + self.bo)
        return Y
        
        
#Download data
(X_train, ytrain), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Rescale to 0 - 1
X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
X_test = (X_test.reshape(-1, 784) / 255).astype(np.float32)

#Add noise to training samples
bitmask = np.random.random((X_train.shape))
bitmask = np.where(bitmask >= 0.6, 0, 1)

#modified training sample with noise
X_train_noise = X_train * bitmask

autoencoder = AutoEncoder(500, '0')
autoencoder.fit(X_train_noise, lr=0.4, epochs=4)

plt.plot(autoencoder.costs)
plt.show()

#Generate some reconstructed output
done = False
while not done:
    rand_idx = np.random.choice(len(X_test))
    rand_sample = X_test[rand_idx]
    x_recon = autoencoder.predict([rand_sample])
    plt.subplot(2,2,1)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Original test image")
    plt.subplot(2,2,2)
    plt.imshow(x_recon.reshape(28,28), cmap='gray')
    plt.title("Reconstructed test image")
    rand_idx = np.random.choice(len(X_train_noise))
    rand_sample = X_train_noise[rand_idx]
    x_recon = autoencoder.predict([rand_sample])
    plt.subplot(2,2,3)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Original noisy train image")
    plt.subplot(2,2,4)
    plt.imshow(x_recon.reshape(28,28), cmap='gray')
    plt.title("Reconstructed train image")
    plt.show()
    
    ans = input("Generate another one y/n?")
    if ans in ['n', 'N']:
        done = True
        

        
        