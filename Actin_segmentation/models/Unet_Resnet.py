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
from .layers.layers import normalize_input, activation_function, regularizer_function, bn_relu_conv2d, bn_relu_conv2dtranspose
        
################################################
# Unet + Resnet
################################################

class Unet_Resnet(CNN_Base):
    """
    Unet + resnet functions
    see https://link.springer.com/chapter/10.1007/978-3-319-46976-8_19
    """
    
    def __init__(self, model_dir = None, **kwargs):       
        super().__init__(model_dir = model_dir, **kwargs)
        
    def bottleneck_block(self, inputs, 
                         upsample = False,
                         filters = 8,
                         strides = 1, dropout_value = None, acti = None, padding = None, 
                         kernel_initializer = None, weight_regularizer = None, name = None):            
        # Bottleneck_block
        with tf.name_scope("Bottleneck_block" + name):
            output = bn_relu_conv2d(inputs, filters, 1,  acti=acti, padding=padding, strides=strides, 
                                    kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
            output = bn_relu_conv2d(output, filters, 3,  acti=acti, padding=padding, 
                                    kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
            
            if upsample == True:
                output = bn_relu_conv2dtranspose(output, filters, (2,2), strides = (2,2), acti=acti, padding=padding, 
                                                kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)
                output = Conv2D(filters * 4, (1,1), padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=regularizer_function(weight_regularizer))(output)
            else:
                output = bn_relu_conv2d(output, filters*4, 1,  acti=acti, padding=padding, 
                                        kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)

            output = Dropout(dropout_value)(output)
            
            # reshape input to the same size as output
            if upsample == True:
                inputs = UpSampling2D()(inputs)
            if strides == 2:
                inputs = Conv2D(output.shape[3].value, 1, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(inputs)
            
            # ensure number of filters are correct between input and output
            if output.shape[3] != inputs.shape[3]:
                inputs = Conv2D(output.shape[3].value, 1, padding=padding, kernel_initializer=kernel_initializer)(inputs)

            return Add()([output, inputs])
        
    def simple_block(self, inputs, filters,
                     strides = 1, dropout_value = None, acti = None, padding = None, 
                     kernel_initializer = None, weight_regularizer = None, name = None):
            
        with tf.name_scope("Simple_block" + name):
            output = BatchNormalization()(inputs)
            output = activation_function(output, acti)
            output = MaxPooling2D()(output)
            output = Conv2D(filters, 3, padding=padding, strides=strides,
                            kernel_initializer=kernel_initializer, 
                            kernel_regularizer=regularizer_function(weight_regularizer))(output)

            output = Dropout(dropout_value)(output)

            inputs = Conv2D(output.shape[3].value, 1, padding=padding, strides=2, kernel_initializer=kernel_initializer)(inputs)
            
            return Add()([output, inputs])
        
    def simple_block_up(self, inputs, filters,
                        strides = 1, dropout_value = None, acti = None, padding = None, 
                        kernel_initializer = None, weight_regularizer = None, name = None):
        
        with tf.name_scope("Simple_block_up" + name):
            output = bn_relu_conv2d(inputs, filters, 3,  acti=acti, padding=padding, strides=strides, 
                                    kernel_initializer=kernel_initializer, weight_regularizer=weight_regularizer)

            output = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer)(output)

            output = Dropout(dropout_value)(output)
            
            inputs = UpSampling2D()(inputs)
            inputs = Conv2D(output.shape[3].value, 1, padding=padding, kernel_initializer=kernel_initializer)(inputs)

            return Add()([output, inputs])
    

    def build_model(self, unet_input, mean_std_normalization = None, 
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
        inputs = Input(unet_input)
        
        # normalize images
        layer = normalize_input(inputs, 
                                scale_input = self.config.get_parameter("scale_input"),
                                mean_std_normalization = self.config.get_parameter("mean_std_normalization"),
                                mean = mean, std = std)

        # encoder arm
        layer_1 = Conv2D(filters, (3, 3), padding = padding, 
                         kernel_initializer = kernel_initializer, 
                         kernel_regularizer = regularizer_function(weight_regularizer), name="Conv_layer_1")(layer)
        
        layer_2 = self.simple_block(layer_1, filters, 
                                    dropout_value = dropout_value, acti = acti, padding = padding, 
                                    kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                    name = "_layer_2")
        
        layer = layer_2
        layer_store = [layer]
        
        for i, conv_layer_i in enumerate(self.config.get_parameter("bottleneck_block"), 1):
            strides = 2
            
            # last layer of encoding arm is treated as across    
            if i == len(self.config.get_parameter("bottleneck_block")):
                layer = self.bottleneck_block(layer, filters = filters, 
                                              strides = strides, dropout_value = dropout_value, acti = acti, padding = padding, 
                                              kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                              name = "_layer_{}".format(2 + i))

                for count in range(conv_layer_i-2):
                    layer = self.bottleneck_block(layer, filters = filters, 
                                                  dropout_value = dropout_value, acti = acti, padding = padding, 
                                                  kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                                  name="_layer_{}-{}".format(2 + i, count))
                    
                layer = self.bottleneck_block(layer, upsample = True,
                                              filters = filters, strides = 1,
                                              dropout_value = dropout_value, acti = acti, padding = padding, 
                                              kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                              name = "_up_layer_{}".format(2 + i))
            else:       
                layer = self.bottleneck_block(layer, filters = filters, 
                                              strides = strides, dropout_value = dropout_value, acti = acti, padding = padding, 
                                              kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                              name = "_layer_{}".format(2 + i))

                for count in range(conv_layer_i - 1):
                    layer = self.bottleneck_block(layer, filters = filters, 
                                                  dropout_value = dropout_value, acti = acti, padding = padding, 
                                                  kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                                  name="_layer_{}-{}".format(2 + i, count))
                filters = filters*2
                layer_store.append(layer)

        # decoder arm
        for i, conv_layer_i in enumerate(self.config.get_parameter("bottleneck_block")[-2::-1], 1):
            filters = filters//2  

            # note that i should be positive possibly due to the way keras/tf model compile works
            layer = Concatenate(axis=3, name="Concatenate_layer_{}".format(i+6))([layer_store[-i], layer])
            
            for count in range(conv_layer_i - 1):
                layer = self.bottleneck_block(layer, filters = filters, 
                                              dropout_value = dropout_value, acti = acti, padding = padding, 
                                              kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                              name="_layer_{}-{}".format(i+6, count))

            layer = self.bottleneck_block(layer, upsample = True,
                                          filters = filters, strides = 1,
                                          dropout_value = dropout_value, acti = acti, padding = padding, 
                                          kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                          name = "_layer_{}".format(i+6))
        
        layer_13 = Concatenate(axis=3, name="Concatenate_layer_13")([layer, layer_2])
        layer_14 = self.simple_block_up(layer_13, filters,
                                        dropout_value = dropout_value, acti = acti, padding = padding, 
                                        kernel_initializer = kernel_initializer, weight_regularizer = weight_regularizer, 
                                        name = "_layer_14")

        layer_15 = Concatenate(axis=3, name="Concatenate_layer_15")([layer_14, layer_1])
        
        layer_16 = Conv2D(filters, (3, 3), padding = padding, 
                          kernel_initializer = kernel_initializer, kernel_regularizer = regularizer_function(weight_regularizer), 
                          name="Conv_layer_16")(layer_15)
        
        layer_17 = BatchNormalization()(layer_16)
        layer_18 = activation_function(layer_17, acti)

        outputs = Conv2D(1, (1, 1), activation = self.config.get_parameter("final_activation"))(layer_18)
        
        return Model(inputs=[inputs], outputs=[outputs], name = self.config.get_parameter('name'))
    
class Unet_Resnet101(Unet_Resnet):
    def __init__(self, model_dir = None, name = 'Unet_Resnet101', **kwargs):
        super().__init__(model_dir = model_dir, **kwargs)
        
        self.config.update_parameter(["model","name"], name)
        self.config.update_parameter(["model","bottleneck_block"], (3, 4, 23, 3))

        # store parameters for ease of use (may need to remove in the future)
        self.conv_layer = self.config.get_parameter("bottleneck_block")

class Unet_Resnet50(Unet_Resnet):
    def __init__(self, model_dir = None, name = 'Unet_Resnet50', **kwargs):
        super().__init__(model_dir = model_dir, **kwargs)
        
        self.config.update_parameter(["model","name"], name)
        self.config.update_parameter(["model","bottleneck_block"], (3, 4, 6, 3))
        
        # store parameters for ease of use (may need to remove in the future)
        self.conv_layer = self.config.get_parameter("bottleneck_block")
        
class Unet_Resnet_paper(Unet_Resnet):
    def __init__(self, model_dir = None, name = 'Unet_Resnet101', **kwargs):
        """
        see https://arxiv.org/pdf/1608.04117.pdf
        """
        super().__init__(model_dir = model_dir, **kwargs)
        
        self.config.update_parameter(["model","name"], name)
        self.config.update_parameter(["model","bottleneck_block"], (3, 8, 10, 3))

        # store parameters for ease of use (may need to remove in the future)
        self.conv_layer = self.config.get_parameter("bottleneck_block")