# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 07:30:04 2020

@author: cupertino_user
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
    
class DBN(tf.keras.Model):
    def __init__(self, D, hidden_layers_units, K, UnsupervisedModel=RBM):  
        #D is input data dimension
        #hidden_layers_units - list of hidden layer sizes
        #K is number of output categories
        super(DBN, self).__init__(dtype='float32')
        count = 0
        self.hidden_layers=[]
        input_size = D
        for output_units in hidden_layers_units:
            rbm = RBM(input_size, output_units, 'rbm_block%s' % count)
            rbm.compile(tf.keras.optimizers.Adam(0.01))
            self.hidden_layers.append(rbm)
            count += 1
            input_size = output_units
        #Initialize logistic regression layer 
        self.final_rbm_layer = tf.keras.layers.Input(shape=(None, input_size)) 
        self.outputs = tf.keras.layers.Dense(K, activation='softmax')(self.final_rbm_layer )
        # self.init_last_layer(self, D, hidden_layers_units[-1], K)   
        self.costs = []         
    # def init_last_layer(self, D, M, K):
    #     #Initialize logistic regression layer        
    #     final_rbm = tf.Variable(tf.float32, shape=(None, M))      
    #     self.outputs = tf.keras.layers.Dense(K, activation='softmax')(final_rbm)
        
        
    def predict(self, X):
        current_input = X
        #forward ut through all rbm blocks
        for rbm in self.hidden_layers:
           Z = rbm.forward_hidden(current_input)
           current_input = Z
       
        #forward it thru the last regression layer
        return tf.argmax(self.outputs(current_input), 1)
    
    def call(self, inputs, epochs=10, batch_size=100):  
        current_input = inputs[0]
        y = inputs[1]
        for rbm in self.hidden_layers:
            rbm.fit(current_input, epochs=3, batch_size=batch_size)
            current_input = rbm.forward_hidden(current_input)
        final_output = tf.argmax(self.outputs(current_input), 1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, final_output)
        self.add_loss(loss)
        self.costs.append(loss)        
        return final_output
     
#Download data
(Xtrain, ytrain), (Xtest, ytest) = tf.keras.datasets.mnist.load_data()

#Rescale to 0 - 1
Xtrain = (Xtrain.reshape(-1, 784) / 255).astype(np.float32)
Xtest = (Xtest.reshape(-1, 784) / 255).astype(np.float32)

#Add noise to training samples
bitmask = np.random.random((Xtest.shape))
bitmask = np.where(bitmask >= 0.6, 0, 1)

#modified training sample with noise
X_test_noise = (Xtest * bitmask).astype(np.float32)

dbm = DBN(Xtrain.shape[1], [750, 500, 250], len(set(ytrain)))
dbm.compile(tf.keras.optimizers.Adam(0.01))
network_history = dbm.fit(Xtrain, ytrain)

#plot the loss
plt.plot(network_history.history['loss'])
plt.title("DBM training loss")
plt.show()

#Generate some reconstructed output
done = False
while not done:
    rand_idx = np.random.choice(len(Xtest))
    rand_sample = Xtest[rand_idx]
    ypred = dbm.predict(rand_sample.reshape(1, -1))
    plt.subplot(1,2,1)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Y true {0} Y pred {1}".format(ytest[rand_idx], ypred))   
    rand_idx = np.random.choice(len(X_test_noise))
    rand_sample = X_test_noise[rand_idx]
    ypred = dbm.predict(rand_sample.reshape(1, -1))
    plt.subplot(1,2,2)
    plt.imshow(rand_sample.reshape(28,28), cmap='gray')
    plt.title("Noise Y true {0} Y pred {1}".format(ytest[rand_idx], ypred)) 
    plt.tight_layout(True)
    plt.show()
    
    ans = input("Generate another one y/n?")
    if ans in ['n', 'N']:
        done = True    