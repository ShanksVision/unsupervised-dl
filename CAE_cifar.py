#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 07:37:59 2020

@author: sjagadee

Covolutional auto encoder for CIFAR10
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

cifar10_data = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10_data.load_data()

x_train = x_train/255
x_test = x_test/255

#Create the model
input_img = tf.keras.Input(shape=(32,32,3))
x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(input_img)
x = tf.keras.layers.MaxPool2D(padding='same')(x)
x = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPool2D(padding='same')(x)
x = tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu')(x)
encoded = tf.keras.layers.MaxPool2D(padding='same')(x)

xhat = tf.keras.layers.Conv2D(4, 3, padding='same', activation='relu')(encoded)
xhat = tf.keras.layers.UpSampling2D()(xhat)
xhat = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu')(xhat)
xhat = tf.keras.layers.UpSampling2D()(xhat)
xhat = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(xhat)
xhat = tf.keras.layers.UpSampling2D()(xhat)
decoded = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(xhat)

cae_model = tf.keras.models.Model(input_img, decoded)

#Compile the model
cae_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

#review model summary
cae_model.summary()

#fit the model
# cae_model.fit(x=x_train, y=x_train, batch_size=128, epochs=50, 
#               callbacks=[TensorBoard(log_dir='tmp/cae')],
#               validation_data=(x_test, x_test))

cae_model.fit(x=x_train, y=x_train, batch_size=128, epochs=50,               
              validation_data=(x_test, x_test))

#Generate some reconstructed output
done = False
while not done:
    rand_idx = np.random.choice(len(x_test))
    rand_sample = x_test[rand_idx]
    x_recon = cae_model.predict(rand_sample.reshape(1,32,32,3))
    plt.subplot(1,2,1)
    plt.imshow(rand_sample)
    plt.title("Original test image")
    plt.subplot(1,2,2)
    plt.imshow(x_recon.reshape(32,32,3))
    plt.title("Reconstructed test image")
    plt.show()
    
    ans = input("Generate another one y/n?")
    if ans in ['n', 'N']:
        done = True
