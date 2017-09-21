# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:13:03 2017

@author: shahik
"""

 #Importing Keras libraries and Packages
  
 from keras.models import Sequential    
 from keras.layers import Conv2D  
 from keras.layers import MaxPooling2D  
 from keras.layers import Flatten  
 from keras.layers import Dense 
 
 # Initialising the CNN
 classifier= Sequential() 
              
 #classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu')) # input_shape= shape of input image to apply feature detector on, 3d array for colored images,64 is dimension(we using , tensorflow that has riverse order of no's,in theano order is opposite)   , 2d for black and white, images should be in same formate
                            #Updated Version
                classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
 
 classifier.add(MaxPooling2D(pool_size=(2,2)))   
 
  classifier.add(Conv2D(32, (3, 3), activation="relu"))
 
  classifier.add(MaxPooling2D(pool_size=(2,2)))
    
  classifier.add(Flatten())

 classifier.add(Dense(activation="relu", units=128))
 classifier.add(Dense(activation="sigmoid", units=1))                 
                  
 classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', # path to images
                                                 target_size = (64, 64), # Expected Size of images : convolution input_shape=64.  Put 128 for more accuracy and GPU , CPU will take time
                                                 batch_size = 32, 
                                                 class_mode = 'binary') # Two types so binary

test_set = test_datagen.flow_from_directory('dataset/test_set', # Path
                                            target_size = (64, 64), # Put 128 for more accuracy and so on USE GPU or more time
                                            batch_size = 32,
                                            class_mode = 'binary')
# Takes 20 mins to run
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,# no of images in training_set 
                         nb_epoch = 25, # no of epoch
                         validation_data = test_set, # on which we evaluate performance
                         nb_val_samples = 2000) # no of images in test set
 
 
 
  
