import pandas as pd
import numpy as np

import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
import skopt

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
from keras.utils import generic_utils # For progress bar

import keras.backend as K

import skimage

# General support
import math
import tensorflow as tf

# For plots
import matplotlib.pyplot as plt

# Dimensionality reduction
#import umap
from sklearn.manifold import TSNE

import copy

from sklearn.metrics import roc_auc_score


#prepare dataset

import pandas as pd
path = os.path.join('/scratch/smp/s4396046/project', 'NvAndMelNoDuplicates.zip')

X_df = pd.read_pickle(path)
print("X_df.shape =", X_df.shape)
print("Inspect X_df", os.linesep, X_df.head())

# Store number of images in the dataset
num_images = X_df.shape[0]
num_classes = 2
print(num_images)

# Get y, a vector containing the classes for each image
y = X_df.pop('id')

# Allocate space for all images
X = np.empty((num_images, 224, 224, 3), dtype='float32') #16 bit breaks things

for i in range(0, num_images-1):
    X[i, :, :, :] = np.reshape(X_df['image'][i], (224, 224, 3)) / 255

print(y[:].value_counts())

print("X.shape =", X.shape)
print("y.shape =", y.shape)

del X_df
print('here2')
# generate 4x the melanoma data with augmentation
# include original + horizontal + vertical + horizontal and vertical
# for each angle, optionally a) zoom in, b) change saturation/brightness, 
# c) add noise/blurring

# get all the X images that are melanomas
mel_X = X[np.where(y == "mel")[0]]

print(mel_X.shape)

# for each one
aug_data = []
for i in range(mel_X.shape[0]):
  im = mel_X[i, :, :, :]
  aug_im = copy.deepcopy(im)
  del(im)
  # for each flipped version
  flip = np.random.choice(['horizontal', 'vertical', 'both'])
  if flip == 'horizontal':
    aug_im = np.flipud(aug_im)
  elif flip == 'vertical': 
    aug_im = np.fliplr(aug_im)
  elif flip == 'both':
     aug_im = np.flipud(np.fliplr(aug_im))

    # do the augmentation
  gamma = np.random.uniform(0.5, 2.5)
  aug_im = skimage.exposure.adjust_gamma(aug_im, gamma, gain=1)


  aug_data.append(copy.deepcopy(aug_im))

  del(aug_im)


aug_len = len(aug_data)

X = np.append(X, np.asarray(aug_data), axis = 0)
del(aug_data)

len(X)

for i in range(aug_len):
  y = np.append(y, 'mel')

# separate data into validation and training sets
# randomly sample n_samples of class nev and class mel, collect into sets 
# of n_trials for training the models


#30 models were returned - this file was ran twice in parallel (Random sampling for the data seeds)
n_trials = 15
#n_epochs = 10

try:
  y = y.values
except:
  pass

nsamples = y[y == "mel"].shape[0] * 2
print("subsampling to", nsamples)
  
data_splits = {}
for seed in range(n_trials):
  np.random.seed(seed)
  
  # randomly sample equally from each class
  class0 = np.where(y == "nv")[0]
  class1 = np.where(y == "mel")[0]
  
  class0 = np.random.choice(class0, nsamples//2, replace=False)
  class1 = np.random.choice(class1, nsamples//2, replace=False)
  # print("class0", class0[:5])
  
  nsplit = int(nsamples//2 * 0.5)
  
  np.random.shuffle(class0)
  np.random.shuffle(class1)
  
  class0_train = class0[:nsplit]
  class0_test = class0[nsplit:]
  class1_train = class1[:nsplit]
  class1_test = class1[nsplit:]
  train_idxs = np.append(class0_train, class1_train)
  test_idxs = np.append(class0_test, class1_test)
  np.random.shuffle(train_idxs)
  np.random.shuffle(test_idxs)
  
  X_train = X[train_idxs]
  y_train = y[train_idxs]
  X_test = X[test_idxs]
  y_test = y[test_idxs]
  
  
  # this converts the 'mel' and 'nv' classes to 0s and 1s
  y_train[y_train == "nv"] = 0
  y_train[y_train == "mel"] = 1
  y_train = np.concatenate((np.expand_dims(y_train, -1), np.expand_dims(1 - y_train, -1)), axis=1)

  y_test[y_test == "nv"] = 0
  y_test[y_test == "mel"] = 1
  y_test = np.concatenate((np.expand_dims(y_test, -1), np.expand_dims(1 - y_test, -1)), axis=1)
  # print(y_train.shape, y_test.shape)
  
  
  data_splits[seed] = [X_train, y_train, X_test, y_test]





np.save("/scratch/smp/s4396046/project/trial_0_y_test.npy", data_splits[0][3], allow_pickle=True)

"""**4. Build classifier**"""

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# This defines the hyperparameter search space, add more parameters here
# as you see fit :)
space = [Real(1e-6, 0.01, "log-uniform", name='learning_rate'),
          Real(0.1, 0.8, name='dropout'),
          Real(0.8, 1.0, name='momentum'),
          Real(0.9, 1.0, name='beta_1'),
          Real(0.99, 1.0, name='beta_2'),
          Integer(low=5,high=20, name = 'epochs'),
          Integer(low=50, high=225, name='num_dense_nodes'),
          Categorical(categories=['SGD', 'Adam'],
                             name='optimizer_type')
          ]

# Define a Callback class that stops training once accuracy reaches 90%
#class myCallback(tf.keras.callbacks.Callback):
  #def on_epoch_end(self, epoch, logs={}):
   # if(logs.get('acc')>0.87):
      #print("\nReached 90% accuracy so cancelling training!")
      #self.model.stop_training = True



# make the model - isolating this as a function to be called for each HP search
# iteration to pass in the updated parameters

def make_model(learning_rate, dropout, momentum, beta_1, beta_2,
               num_dense_nodes, optimizer_type):
  # Create inception model with our input shape
  base_model = InceptionV3(weights='imagenet',
          input_shape=(224, 224, 3), include_top=False)

  # Add extra dense layers
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(num_dense_nodes, activation='relu', kernel_initializer='he_normal')(x)
  x = Dropout(rate=dropout)(x)
  predictions = Dense(num_classes, activation='sigmoid')(x)
  model = Model(inputs=base_model.input, outputs=predictions)

  if optimizer_type == "Adam":
    optimizer = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2)
  elif optimizer_type == "SGD":
    optimizer = SGD(lr=learning_rate, momentum=momentum)

  model.compile(loss='binary_crossentropy',
          optimizer=optimizer,
          metrics=['accuracy'])
  return model





batch_size = 32
best_accuracy = {} 
for seed in range(n_trials):
  best_accuracy[seed] = 0.0


for seed in range(n_trials):
  print('We are currently training on seed:', seed) 
  # for each iteration of the hyperparameter search, return a set of parameters
  # and feed them into the relevant parts
  # run training of the model for this seed, save with seed num
  X_train, y_train, X_test, y_test = data_splits[seed]
  path_best_model = 'inception_saved_trial_{}.h5'.format(seed)
  
  @use_named_args(dimensions=space)
  def fitness(learning_rate, dropout, momentum, beta_1, beta_2,
               num_dense_nodes, optimizer_type, epochs):

      # Print the hyper-parameters.
      print('learning rate: {0:.1e}'.format(learning_rate))
      print('num_dense_nodes:', num_dense_nodes)
      print('dropout:', dropout)
      print('optimizer_type:', optimizer_type)
      print('epochs:', epochs)

      # Create the neural network with these hyper-parameters.
      model = make_model(learning_rate=learning_rate, 
                         dropout=dropout, 
                         momentum=momentum, 
                         beta_1=beta_1, beta_2=beta_2,
                         num_dense_nodes=num_dense_nodes, 
                         optimizer_type=optimizer_type)

      # Use Keras to train the model.
      history = model.fit(x=X_train,
                          y=y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data= (X_test,y_test))

      # Get the classification accuracy on the validation-set
      # after the last training-epoch.
      accuracy = history.history['val_acc'][-1]
      # auc_val = history.history['val_auc'][-1]

      # Print the classification accuracy.
      print()
      print("Accuracy: {0:.2%}".format(accuracy))
      print()

      # Save the model if it improves on the best-found performance.
      # We use the global keyword so we update the variable outside
      # of this function.
      global best_accuracy

      if accuracy > best_accuracy[seed]:
          # Save the new model to harddisk.
          model_path = os.path.join('/scratch/smp/s4396046/project', path_best_model)
          model.save(model_path)
	

          # Update the classification accuracy.
          best_accuracy[seed] = accuracy
          # best_auc = auc_val
         

      # Delete the Keras model with these hyper-parameters from memory.
      del model

      K.clear_session()

      return -accuracy
  
  search_result = gp_minimize(func=fitness,
                            dimensions=space,
                            acq_func='EI', # Expected Improvement.
                            n_calls=30,
			    n_random_starts = 10,
                            verbose = True)
  print('Seed: ',seed)
  print("BEST ACCURACY: ", best_accuracy)
  #space_res = search_result.space
  print('hyper_params ', search_result.x)
