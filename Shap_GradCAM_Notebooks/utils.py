#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:14:39 2019

@author: Kyle
"""

# Keras is used for loading the model
import keras
from keras.models import Model
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

from keras import activations
# and also has a utility function for converting from diagnosis to one-hot encoding
# (i.e. two entries for each class, 1 indicates positive for that class and 0 indicates
# negative for that class)
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils import generic_utils
from vis.utils import utils
from vis.utils import utils
from scipy import *


# We also need sys, os and copy for some processes to retrieve the data (the test set
# and model) as well as to install some new modules in colab
import sys
import os
import copy

# Importing matplotlib, numpy and pandas for handling the test set data and 
# later for displaying the original image + explanations created by SHAP, Gradcam
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

# An image segmentation algorithm we will use for SHAP
from skimage.segmentation import slic

# Importing the Pillow Imaging Library to handle image files
import PIL
from PIL import Image

import shap
from vis import visualization



'''
A function that takes arguments detailing the segments in the image, the segments
that need to be hidden to use SHAP, the image itself and an optional background color.

This function will eventually be used by shap to determine the effect that "missing"
segments have on the final likelihood results.

Arguments:

  zs (2d int[]): A two-dimensional array indicating whether a segment is "hidden" (replaced
                 by an average color) for several instances of the same image.
  segmentation (2d int[]): A two-dimensional array indicating which pixels are part of which
                           segments in the image.
  image (3d int[]): The image being used, in numpy array format.
  background (1d int[]): The optional background color being used.
  
Returns:
  
  (4d int[]): A 4-dimensional array containing multiple instances of the same image
              with different combinations of segments hidden.
'''
def mask_image(zs, segmentation, image, background=None):
    # If we have no set background color, just take the mean color from the image by taking
    # The mean across the x and y (0 and 1) axes
    if background is None:
        background = image.mean((0,1))
    
    # Create an empty array of images with the same size as the image argument
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    
    # For each image
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        # Copy the image to the empty array of images
        # For each segment in the zs array:
        for j in range(zs.shape[1]):
            # If the segment has a value of 0,
            if zs[i,j] == 0:
                # "Hide" the segment by changing the pixel colors
                # to a background color
                out[i][segmentation == j,:] = background         
    return out


'''
A function that an array of values and an array of segment locations and returns
an array of the same size as the segmentation array, but with the value array elements
in place of segment numbers

Arguments:

  values (1d int[]): An array of integers representing particular values for particular
                     segments
  segmentation (2d int[]): A two-dimensional array indicating which pixels are part of which
                           segments in the image.
                           
  Returns:
  
  (2d int[]): A 2-dimensional indicating the value for that pixel (which will be identical
              throughout the segment that pixel is a part of).

'''
def fill_segmentation(values, segmentation):
    # Create an empty array with the same shape as the segmentation argument
    out = np.zeros(segmentation.shape)
    # For each segment,
    for i in range(len(values)):
        # Replace coordinates in that segment with the value for that segment
        out[segmentation == i] = values[i]
    return out


'''
The below function takes a particular index (referring to an image in the test set),
along with the model we imported above, a "mode" (which is simply whether the shap values represent 
the melanoma or nevi prediction) and an optional ability to save the resultant image
of the explanation.

The function also calculates the GradCAM explanation for that image and displays both
explanations side-by-side with the original image.

Arguments:

  index (int): Index of image in the test set
  model (keras.Model): A convolutional neural network model object.
  mode (int): 0 if Melanoma has been predicted, 1 if Nevus has been predicted
  save (boolean): An optional argument to save the image to the current directory.
  
'''
#TODO Modify to take an image as input rather than using the index, for the purposes of iterating over all images an index is kinda better... for now...

def Shap_single(image, model, gmodel, save = False, filename = None, relative_path = None):    
    # Retrieve the image from the test set
    image = image
    
    # Create a function to return the prediction based on an array that indicates which
    # segments are to be hidden
    def f(z):
        #The below line is no longer used, but is kept there in case we ever 
        #switch to using native inception preprocessing, or we implement preprocessing functions
        #return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))
        
        mask = mask_image(z, segments_slic, img_orig, 255)
        mask /= 255
        return model.predict(mask)
        
    # Create a dictionary
    
    feature_names = {'0': ['n01', 'mel'],
                   '1': ['n02', 'nev']}
    
    # Create a numpy array of pixel data for the image
    img = Image.fromarray(image)
    # And save the image as img_orig
    img_orig = image
    
    # Generate segments from the image
    segments_slic = slic(img, compactness=3, sigma=13, max_size_factor = 3, 
                         min_size_factor = 0.5, convert2lab = True,
                         enforce_connectivity = True)
   
  
    # Get the number of segments by calculating the number of unique segment numbers
    # in the array returned just above
    num_seg = len(np.unique(segments_slic))
    
    # Use Kernel SHAP to explain the model's predictions
    # First create an explainer
    explainer = shap.KernelExplainer(f, np.zeros((1,num_seg)))
    # Then generate an explanation using the explainer
    shap_values = explainer.shap_values(np.ones((1,num_seg)), nsamples=2500) # runs VGG16 (this model) 2000 times
    
 
    # Get the top predictions from the model
    preds = model.predict(np.expand_dims(img_orig.copy() / 255, axis=0))
    
    print(preds)
    top_preds = np.argsort(-preds)
    inds = top_preds[0]
    #reverse this inequallity for 2nd prediction
    if preds[0][0] > preds[0][1]:
      mode = 0
    else:
      mode = 1
    
    # Create a custom colormap for shap (blue and red for positive and negative
    # contributions respectively)
    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    # For negative values, create 100 red colors that vary based
    # on opacity (i.e. the closer to 0, the more transparent the red)
    for l in np.linspace(1,0,100):
        colors.append((245/255,39/255,87/255,l))
    # Do the same for positive contributions (blue)
    for l in np.linspace(0,1,100):
        colors.append((24/255,196/255,93/255,l))
    cm = LinearSegmentedColormap.from_list("shap", colors)
    
    
    # Get the max SHAP value for this image
    max_val = np.max([np.max(np.abs(shap_values[i][:,:-1])) for i in range(len(shap_values))])
    
    def full_frame(width=None, height=None):
      import matplotlib as mpl
      mpl.rcParams['savefig.pad_inches'] = 0
      figsize = None if width is None else (width, height)
      fig = pl.figure(figsize=figsize)
      ax = pl.axes([0,0,1,1], frameon=False)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      pl.autoscale(tight=True)
    
    #SHAP
    pl.figure()
    pl.tight_layout()
    pl.axis('off')
    full_frame()
    #[inds[0]][0] for top prediction, [inds[1]][0] for second prediction
    m = fill_segmentation(shap_values[inds[0]][0], segments_slic)
    pl.imshow(img, alpha=0.8)
    pl.imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
    
    if save: 
      f_name = f"id_{filename}_SHAP"
      #'Explanations_TEST/'
      f_path = relative_path+f_name
      pl.savefig(f_path, bbox_inches = 'tight',pad_inches = 0)
   
      
    #GRAD-CAM
    pl.figure()
    pl.tight_layout()
    full_frame()
    # Get the original image and normalize each rgb value to between 0 and 1
    # Utility to search for layer index by name
    layer_idx = utils.find_layer_idx(gmodel,'dense_2')
    image = image.astype('float64') / 255
    gradcam = visualization.visualize_cam(gmodel, layer_idx, mode, image, penultimate_layer_idx=310)
    pl.axis('off')
    pl.imshow(image)
    pl.imshow(gradcam, cmap='jet', alpha=0.5)
    
    
    if save: 
      f_name = f"id_{filename}_Grad"
      f_path = relative_path+f_name
      pl.savefig(f_path, bbox_inches = 'tight' ,pad_inches = 0)
      
      


#batch shap
# Automatically produces all images in x_test     
def batch_shap(data, model, gmodel, path):
    for i in range(len(data)): 
       filename = str(i)      
       Shap_single(data[i], model, gmodel, save = True, filename = filename, relative_path = path )
# Shouldn't need to use this --- it was just for when i needed to produce images with id's but not all of them...
#If that makes sense        
def batch_shap2(data, id_list, model, gmodel ,path):
    for i in id_list:
        filename = str(i)
        Shap_single(data[i], model, gmodel, save = True, filename = filename, relative_path = path)