# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:51:57 2020

@author: shankar.j

Convolutional auto encoders on a real world application
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow.keras.preprocessing.image as imageprocess 
               
def preprocess_input(x):
    return x/255            
                   
train_path = '../Data/B1/train'
test_path = '../Data/B1/test'
val_path = '../Data/B1/val'

#Preprocessed image size
#image_shape = (440, 852, 1)    
image_shape = (424, 220, 1)       

#Get train, test file path lists and class counts
train_imgs = glob.glob(train_path + '/*/*.png')
test_imgs = glob.glob(test_path + '/*/*.png')
val_imgs = glob.glob(val_path + '/*/*.png')

#create a l2 regularizer
reg = tf.keras.regularizers.l1()

#Enable multi-gpu model
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    
    #Create the model
    input_img = tf.keras.Input(shape=image_shape)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=None)(input_img)
    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=None)(x)                               
    x = tf.keras.layers.MaxPool2D(padding='same')(x)
    x = tf.keras.layers.Cropping2D(cropping=((0,0),(0,1)))(x)   
    x = tf.keras.layers.Conv2D(50, 3, padding='same', activation='relu', kernel_regularizer=None)(x)
    encoded = tf.keras.layers.MaxPool2D(padding='same')(x)
    
    xhat = tf.keras.layers.Conv2D(50, 3, padding='same', activation='relu', kernel_regularizer=None)(encoded)
    xhat = tf.keras.layers.UpSampling2D()(xhat)
    xhat = tf.keras.layers.ZeroPadding2D(padding=((0,0),(0,1)))(xhat)   
    xhat = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=None)(xhat)
    xhat = tf.keras.layers.UpSampling2D()(xhat)
    xhat = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=None)(xhat)
    xhat = tf.keras.layers.UpSampling2D()(xhat)
    decoded = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(xhat)
    
    cae_model = tf.keras.models.Model(input_img, decoded)
    
    #Compile the model
    cae_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mean_squared_error')

cae_model.summary()

#Configure the image generator
batch_size = 30

train_gen = imageprocess.ImageDataGenerator(preprocessing_function=preprocess_input)  

train_flow = train_gen.flow_from_directory(train_path, target_size=image_shape[0:2],
                                           color_mode='grayscale', batch_size=batch_size,
                                           class_mode='input')
test_flow = train_gen.flow_from_directory(test_path, target_size=image_shape[0:2],
                                          color_mode='grayscale', batch_size=batch_size,
                                          class_mode='input')
val_flow = train_gen.flow_from_directory(val_path, target_size=image_shape[0:2],
                                         color_mode='grayscale', batch_size=batch_size,
                                         class_mode='input')

#Create numpy arrays associated with train and test images and figure out their shapes
train_img_count = len(train_imgs)
test_img_count = len(test_imgs)
val_img_count = len(val_imgs)



#Fit the model
network_history = cae_model.fit(train_flow, epochs=40, 
                            steps_per_epoch=int(np.ceil(train_img_count/batch_size)) 
                           )

x_test = np.zeros(tuple([test_img_count] + list(image_shape)))
x_train = np.zeros(tuple([train_img_count] + list(image_shape)))
x_val = np.zeros(tuple([val_img_count] + list(image_shape)))

#Populate the train and test numpy arrayss
i = 0
for X, y in test_flow:
    batch_samples = len(y)
    x_test[i:i+batch_samples] = X    
    i += batch_samples
    if(i >= test_img_count):
        break    

i = 0
for X, y in train_flow:
    batch_samples = len(y)    
    x_train[i:i+batch_samples] = X 
    i += batch_samples 
    if(i >= train_img_count):
        break      
    
i = 0
for X, y in val_flow:
    batch_samples = len(y)    
    x_val[i:i+batch_samples] = X 
    i += batch_samples 
    if(i >= val_img_count):
        break 

#plot loss curves and missed predictions
plt.plot(network_history.history['loss'])
plt.title('Loss vs epoch')
plt.show()

def stretch_array_minmax(input):
    max_value = input.max()
    min_value = input.min()
    range = max_value-min_value
    output = (input - min_value)/range
    return output

#Generate some reconstructed output
done = False
while not done:
    rand_idx = np.random.choice(len(x_test))
    rand_sample = x_test[rand_idx]
    x_recon = cae_model.predict(rand_sample.reshape(tuple([1] + list(image_shape))))
    rand_sample = rand_sample.reshape(image_shape[0], image_shape[1])
    x_recon = x_recon.reshape(image_shape[0], image_shape[1])
    diff = stretch_array_minmax(np.abs(rand_sample-x_recon))
    plt.subplot(2,3,1)
    plt.imshow(rand_sample, cmap='gray')
    plt.title("True NG")
    plt.subplot(2,3,2)
    plt.imshow(x_recon, cmap='gray')
    plt.title("Predicted NG")
    plt.subplot(2,3,3)
    plt.imshow(diff, cmap='viridis')
    plt.title("Deviation NG")
    
    rand_idx = np.random.choice(len(x_val))
    rand_sample = x_val[rand_idx]
    x_recon = cae_model.predict(rand_sample.reshape(tuple([1] + list(image_shape))))
    rand_sample = rand_sample.reshape(image_shape[0], image_shape[1])
    x_recon = x_recon.reshape(image_shape[0], image_shape[1])
    diff = stretch_array_minmax(np.abs(rand_sample-x_recon))
    plt.subplot(2,3,4)
    plt.imshow(rand_sample, cmap='gray')
    plt.title("Original OK")
    plt.subplot(2,3,5)
    plt.imshow(x_recon, cmap='gray')
    plt.title("Predicted OK")
    plt.subplot(2,3,6)
    plt.imshow(diff, cmap='viridis')
    plt.title("Deviation OK")
    plt.tight_layout(True)
    plt.show()
    
    ans = input("Generate another one y/n?")
    if ans in ['n', 'N']:
        done = True


          