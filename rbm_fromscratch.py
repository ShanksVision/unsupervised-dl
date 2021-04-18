# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:42:39 2020

@author: Shankar j

Recreated from original code by lazy programmer for practicing purposes
# https://www.udemy.com/unsupervised-deep-learning-in-python
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class RBM(tf.keras.Model):
    def __init__(self, D, M, layer_name):
        #D is the visible units and M is the hidden units
        super(RBM, self).__init__(dtype='float32')
        # super._autocast = True
        self.D = D
        self.M = M
        self.init_params(D, M)
        self.costs = []
        
    def init_params(self, D, M):
        #3 main parameters we need for RBM
        self.W = tf.Variable(tf.random.normal(shape=(D, M)) * np.sqrt(2.0/M))
        self.c = tf.Variable(tf.zeros(M), dtype='float32')
        self.b = tf.Variable(tf.zeros(D), dtype='float32')   
    
        
    def free_energy(self, V):
        b = tf.reshape(self.b, (self.D, 1))
        first_term = -tf.matmul(V, b)
        first_term = tf.reshape(first_term, (-1,))

        second_term = -tf.reduce_sum(tf.nn.softplus(tf.matmul(V, self.W) + self.c),            
                                    axis=1)
        
        return first_term + second_term
    
    def forward_hidden(self, X):
        return tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)
        
    def forward_logits(self, X):
        Z = self.forward_hidden(X)
        return tf.matmul(Z, tf.transpose(self.W)) + self.b
    
    def forward_output(self, X):
        return tf.nn.sigmoid(self.forward_logits(X))   
    
    
    def call(self, inputs, epochs=1):        
        V = inputs
        self.p_h_given_v = tf.nn.sigmoid(tf.matmul(V, self.W) + self.c)
        r = tf.random.uniform(shape=tf.shape(self.p_h_given_v))
        H = tf.cast(r < self.p_h_given_v, tf.float32)
        
        self.p_v_given_h = tf.nn.sigmoid(tf.matmul(H, tf.transpose(self.W)) + self.b)
        r = tf.random.uniform(shape=tf.shape(self.p_v_given_h))
        X_sample = tf.cast(r < self.p_v_given_h, tf.float32)
        
        #define objectve function
        #Reduce free energy F(v) - F(v')
        objective = tf.reduce_mean(self.free_energy(inputs)) - tf.reduce_mean(
            self.free_energy(X_sample))        
        self.add_loss(objective)
        
        #define cost just for observational purposes
        #not used in actual training
        #logits =  self.forward_logits(inputs)
        # cost = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(inputs, logits))
        # self.costs.append(cost)
        return self.forward_output(inputs)  
     
#Download data
(X_train, ytrain), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Rescale to 0 - 1
X_train = (X_train.reshape(-1, 784) / 255).astype(np.float32)
X_test = (X_test.reshape(-1, 784) / 255).astype(np.float32)

#Add noise to training samples
bitmask = np.random.random((X_train.shape))
bitmask = np.where(bitmask >= 0.6, 0, 1)

#modified training sample with noise
X_train_noise = (X_train * bitmask).astype(np.float32)
#X_train_noise = X_train

rbm = RBM(D=X_train.shape[1], M=100, layer_name='rbm-layer')
rbm.compile(tf.keras.optimizers.Adam(0.01))
network_history = rbm.fit(X_train, epochs=10, batch_size=100)

#plot the loss
plt.plot(network_history.history['loss'])
plt.title("Single RBM training loss")
plt.show()

#Generate some reconstructed output
done = False
while not done:
    rand_idx = np.random.choice(len(X_test))
    rand_sample = X_test[rand_idx]
    x_recon = rbm.forward_output(rand_sample.reshape(1, -1))
    plt.subplot(2,2,1)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Original test image")
    plt.subplot(2,2,2)
    plt.imshow(x_recon.numpy().reshape(28,28), cmap='gray')
    plt.title("Reconstructed test image")
    rand_idx = np.random.choice(len(X_train_noise))
    rand_sample = X_train_noise[rand_idx]
    x_recon = rbm.forward_output(rand_sample.reshape(1, -1))
    plt.subplot(2,2,3)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Original noisy train image")
    plt.subplot(2,2,4)
    plt.imshow(x_recon.numpy().reshape(28,28), cmap='gray')
    plt.title("Reconstructed train image")
    plt.tight_layout(True)
    plt.show()
    
    ans = input("Generate another one y/n?")
    if ans in ['n', 'N']:
        done = True    