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

from .CNN_Base import CNN_Base
from .layers.layers import normalize_input, activation_function, regularizer_function, bn_relu_conv2d
    
######
# Unet
######
class Unet(CNN_Base):
    """
    Unet functions
    see https://www.nature.com/articles/s41592-018-0261-2
    """
    
    def __init__(self, model_dir = None, name = 'Unet', **kwargs):
        super().__init__(model_dir = model_dir, **kwargs)
        
        self.config.update_parameter(["model","name"], name)
        
    def build_model(self, input_size, mean_std_normalization = None, 
                    dropout_value = None, acti = None, padding = None, 
                    kernel_initializer = None, weight_regularizer = None):
        
        ### get parameters from config file ###
        filters = self.config.get_parameter("filters")
        
        if dropout_value is None:
            dropout_value = self.config.get_parameter("dropout_value")
        if acti is None:
            acti = self.config.get_parameter("activation_function")
        if padding is None:
            padding = self.config.get_parameter("padding")
        if kernel_initializer is None:
            kernel_initializer = self.config.get_parameter("initializer")
        if weight_regularizer is None:
            weight_regularizer = self.config.get_parameter("weight_regularizer")
        if mean_std_normalization is None:
            if self.config.get_parameter("mean_std_normalization") == True:
                mean = self.config.get_parameter("mean")
                std = self.config.get_parameter("std")
            else:
                mean = None
                std = None
        
        ### Actual network###
        inputs = Input(input_size)
        
        # normalize images
        layer = normalize_input(inputs, 
                                scale_input = self.config.get_parameter("scale_input"),
                                mean_std_normalization = self.config.get_parameter("mean_std_normalization"),
                                mean = mean, std = std)
        
        layer_store = []
        
        # encoding arm
        for _ in range(self.config.get_parameter("levels")):
            layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                                   kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
            layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                                   kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
            layer_store.append(layer)
            layer = MaxPooling2D((2, 2))(layer)
            
            filters = filters * 2
            
        
        layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                               kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
        layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                               kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
        # decoding arm 
        for i in range(self.config.get_parameter("levels")):
            layer = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(layer)
            
            layer = Concatenate(axis=3)([layer, layer_store[-i -1]])
            filters = filters // 2
            
            layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                                   kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
            layer = bn_relu_conv2d(layer, filters, 3,  acti=acti, padding=padding, strides=strides, 
                                   kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(layer)
        
        return Model(inputs=[inputs], outputs=[outputs], name='Unet')
