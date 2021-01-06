import math

import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation
from keras.layers.core import Lambda, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate, Add
from keras import regularizers
from keras import backend as K

import tensorflow as tf

def activation_function(inputs, acti):
    if isinstance(acti, str):
        return Activation(acti)(inputs)
    else:
        return acti(inputs)

def regularizer_function(weight_regularizer):
    if weight_regularizer == 0 or weight_regularizer == None:
        return None
    else:
        return regularizers.l2(weight_regularizer)
    
def bn_relu_conv2d(inputs, filters, filter_size, 
                    strides = 1, acti = None, padding = None, 
                    kernel_initializer = None, weight_regularizer = None, name = ""):
    output = BatchNormalization()(inputs)
    output = activation_function(output, acti)
    output = Conv2D(filters, (filter_size, filter_size), padding=padding, strides = strides,
                    kernel_initializer=kernel_initializer, 
                    kernel_regularizer=regularizer_function(weight_regularizer))(output)
            
    return output

def bn_relu_conv2dtranspose(inputs, filters, filter_size, 
                            strides = 2, acti = None, padding = None, 
                            kernel_initializer = None, weight_regularizer = None, name = ""):
    output = BatchNormalization()(inputs)
    output = activation_function(output, acti)
    output = Conv2DTranspose(filters, (2, 2), strides=strides, padding=padding, 
                             kernel_initializer=kernel_initializer, 
                             kernel_regularizer=regularizer_function(weight_regularizer))(output)
    return output

def normalize_input(inputs, scale_input = False, mean_std_normalization = False, mean = None, std = None):
    if mean_std_normalization is True:
        print("Using normalization")
        return Lambda(lambda x: (x - mean)/std)(inputs)
    elif scale_input is True:
        print("Not using normalization")
        return Lambda(lambda x: x / 255)(inputs)
    else:
        return inputs
            
    