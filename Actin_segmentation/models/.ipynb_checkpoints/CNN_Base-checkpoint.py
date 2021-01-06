import os

import glob
import datetime

import skimage.io
import numpy as np

import tensorflow as tf

import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, ProgbarLogger

from .internals.image_functions import Image_Functions
from .internals.network_config import Network_Config
from .internals.dataset import Dataset

class CNN_Base(Dataset, Image_Functions):
    def __init__(self, model_dir = None, config_filepath = None, **kwargs):
        """Creates the base neural network class with basic functions
    
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is stored
        config_filepath : `str`, optional
            [Default: None] Filepath to the config file
        **kwargs
            Parameters that are passed to :class:`network_config.Network_Config`

        Attributes
        ----------
        config : :class:`network_config.Network_Config`
            Network_config object containing the config and necessary functions
        """
        
        super().__init__()
        
        self.config = Network_Config(model_dir = model_dir, config_filepath = config_filepath, **kwargs)
        
        self.config.update_parameter(["general", "now"], datetime.datetime.now())
        
        if self.config.get_parameter("use_cpu") is True:
            self.initialize_cpu()
        else:
            self.initialize_gpu()
    
    #######################
    # Logging functions
    #######################
    def init_logs(self):
        """Initiates the parameters required for the log file
        """
        # Directory for training logs
        print(self.config.get_parameter("name"), self.config.get_parameter("now"))
        self.log_dir = os.path.join(self.config.get_parameter("model_dir"), "{}-{:%Y%m%dT%H%M}".format(self.config.get_parameter("name"), self.config.get_parameter("now")))
        
        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}_*epoch*.h5".format(self.config.get_parameter("name"), self.config.get_parameter("now")))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        
    def write_logs(self):
        """Writes the log file
        """
        # Create log_dir if it does not exist
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
            
        # save the parameters used in current run to logs dir
        self.config.write_config(os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}-config.yml".format(self.config.get_parameter("name"), self.config.get_parameter("now"))))
        
    #######################
    # Initialization functions
    #######################
    def summary(self):
        """Summary of the layers in the model
        """
        self.model.summary()
        
    def compile_model(self, optimizer, loss):
        """Compiles model
        
        Parameters
        ----------
        optimizer
            Gradient optimizer used in during the training of the network
        loss
            Loss function of the network
        """
        self.model.compile(optimizer, loss = loss, metrics = self.config.get_parameter("metrics"))

    def initialize_model(self):
        """Initializes the logs, builds the model, and chooses the correct initialization function
        """
        # write parameters to yaml file
        self.init_logs()
        if self.config.get_parameter("for_prediction") is False:
            self.write_logs()
            
        # build model
        self.model = self.build_model(self.config.get_parameter("input_size"))
        
        # save model to yaml file
        if self.config.get_parameter("for_prediction") is False:
            self.config.write_model(self.model, os.path.join(self.log_dir, "{}-{:%Y%m%dT%H%M}-model.yml".format(self.config.get_parameter("name"), self.config.get_parameter("now"))))

        print("{} using single GPU or CPU..".format("Predicting" if self.config.get_parameter("for_prediction") else "Training"))
        self.initialize_model_normal()
            
    def initialize_cpu(self):
        """Sets the session to only use the CPU
        """
        config = tf.ConfigProto(
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
        session = tf.Session(config=config)
        K.set_session(session)   
        
    def initialize_gpu(self):
        """Sets the seesion to use the gpu specified in config file
        """
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.get_parameter("visible_gpu")) # needs to be a string
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.tensorflow_backend.set_session(sess)
    
    def initialize_model_normal(self):
        """Initializes the optimizer and any specified callback functions
        """
        opt = self.optimizer_function()
        self.compile_model(optimizer = opt, loss = self.loss_function(self.config.get_parameter("loss")))
        
        if self.config.get_parameter("for_prediction") == False:
            self.callbacks = [self.model_checkpoint_call(verbose = True)]

            if self.config.get_parameter("use_tensorboard") is True:
                self.callbacks.append(self.tensorboard_call())
                
            if self.config.get_parameter("reduce_LR_on_plateau") is True:
                self.callbacks.append(ReduceLROnPlateau(monitor=self.config.get_parameter("reduce_LR_monitor"),
                                                        factor = self.config.get_parameter("reduce_LR_factor"),
                                                        patience = self.config.get_parameter("reduce_LR_patience"),
                                                        min_lr = self.config.get_parameter("reduce_LR_min_lr"),
                                                        verbose = True))
            
            if self.config.get_parameter("early_stopping") is True:
                self.callbacks.append(EarlyStopping(monitor=self.config.get_parameter("early_stopping_monitor"),
                                                    patience = self.config.get_parameter("early_stopping_patience"),
                                                    min_delta = self.config.get_parameter("early_stopping_min_delta"),
                                                    verbose = True))
                
    #######################
    # Optimizer/Loss functions
    #######################         
    def optimizer_function(self, learning_rate = None):
        """Initialize optimizer function
        
        Parameters
        ----------
        learning_rate : `int`
            Learning rate of the descent algorithm
            
        Returns
        ----------
        optimizer
            Function to call the optimizer
        """
        if learning_rate is None:
            learning_rate = self.config.get_parameter("learning_rate")
        if self.config.get_parameter("optimizer_function") == 'sgd':
            return keras.optimizers.SGD(lr = learning_rate, 
                                        decay = self.config.get_parameter("decay"), 
                                        momentum = self.config.get_parameter("momentum"), 
                                        nesterov = self.config.get_parameter("nesterov"))
        elif self.config.get_parameter("optimizer_function") == 'rmsprop':
            return keras.optimizers.RMSprop(lr = learning_rate, 
                                            decay = self.config.get_parameter("decay"))
        elif self.config.get_parameter("optimizer_function") == 'adam':
            return keras.optimizers.Adam(lr = learning_rate, 
                                         decay = self.config.get_parameter("decay"))
        
    def loss_function(self, loss):
        """Initialize loss function
        
        Parameters
        ----------
        loss : `str`
            Name of the loss function
            
        Returns
        ----------
        loss
            Function to call loss function
        """
        if loss == "binary_crossentropy":
            print("Using binary crossentropy")
            return loss
        elif loss == "jaccard_distance_loss":
            print("Using jaccard distance loss")
            from .internals.losses import jaccard_distance_loss
            return jaccard_distance_loss
        elif loss == "lovasz_hinge":
            print("Using Lovasz-hinge loss")
            from .internals.losses import lovasz_loss
            return lovasz_loss
        elif loss == "dice_loss":
            print("Using Dice loss")
            from .internals.losses import dice_coef_loss
            return dice_coef_loss
        elif loss == "bce_dice_loss":
            print("Using 1 - Dice + BCE loss")
            from .internals.losses import bce_dice_loss
            return bce_dice_loss
        elif loss == "ssim_loss":
            print("Using DSSIM loss")
            from .internals.losses import DSSIM_loss
            return DSSIM_loss
        elif loss == "bce_ssim_loss":
            print("Using BCE + DSSIM loss")
            from .internals.losses import bce_ssim_loss
            return bce_ssim_loss
        elif loss == "mean_squared_error":
            return keras.losses.mean_squared_error
        elif loss == "mean_absolute_error":
            return keras.losses.mean_absolute_error
        elif loss == "ssim_mae_loss":
            print("Using DSSIM + MAE loss")
            from .internals.losses import dssim_mae_loss
            return dssim_mae_loss
        else:
            print("Using {}".format(loss))
            return loss
        
    #######################
    # Callbacks
    #######################     
    def tensorboard_call(self):
        """Initialize tensorboard call
        """
        return TensorBoard(log_dir=self.log_dir, 
                           batch_size = self.config.get_parameter("batch_size_per_GPU"), 
                           write_graph=self.config.get_parameter("write_graph"),
                           write_images=self.config.get_parameter("write_images"), 
                           write_grads=self.config.get_parameter("write_grads"), 
                           update_freq='epoch', 
                           histogram_freq=self.config.get_parameter("histogram_freq"))
    
    def model_checkpoint_call(self, verbose = 0):
        """Initialize model checkpoint call
        """
        return ModelCheckpoint(self.checkpoint_path, save_weights_only=True, verbose=verbose)
    
    #######################
    # Clear memory once training is done
    #######################
    def end_training(self):
        """Deletes model and releases gpu memory held by tensorflow
        """
        # del reference to model
        del self.model
        
        # clear memory
        tf.reset_default_graph()
        K.clear_session()
        
        # take hold of cuda device to shut it down
        from numba import cuda
        cuda.select_device(0)
        cuda.close()
    
    #######################
    # Train Model
    #######################
    def train_model(self, verbose = True):
        """Trains model
        
        Parameters
        ----------
        verbose : `int`, optional
            [Default: True] Verbose output
        """      
        history = self.model.fit(self.aug_images, self.aug_ground_truth, validation_split = self.config.get_parameter("val_split"),
                                 batch_size = self.config.get_parameter("batch_size"), epochs = self.config.get_parameter("num_epochs"), shuffle = True,
                                 callbacks=self.callbacks, verbose=verbose)
        
        self.end_training()
        
    #######################
    # Predict using loaded model weights
    ####################### 
    # TODO: change to load model from yaml file
    def load_model(self, model_dir = None):
        """Loads model from h5 file
        
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Directory containing the model file
        """
        # TODO: rewrite to load model from yaml file
        if model_dir is None:
            model_dir = self.config.get_parameter("model_dir")
            
        if os.path.isdir(model_dir) is True:
            list_weights_files = glob.glob(os.path.join(model_dir,'*.h5'))
            list_weights_files.sort() # To ensure that [-1] gives the last file
            
            model_dir = os.path.join(model_dir,list_weights_files[-1])

        self.model.load_model(model_dir)
        print("Loaded model from: " + model_dir)
        
    def load_weights(self, model_dir = None, weights_index = -1):
        """Loads weights from h5 file
        
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Directory containing the weights file
        weights_index : `int`, optional
            [Default: -1] 
        """
        if model_dir is None:
            model_dir = self.config.get_parameter("model_dir")
        
        if os.path.isdir(model_dir) is True:
            list_weights_files = glob.glob(os.path.join(model_dir,'*.h5'))
            list_weights_files.sort() # To ensure that [-1] gives the last file
            self.weights_path = list_weights_files[weights_index]
            model_dir = os.path.join(model_dir, self.weights_path)
        else:
            self.weights_path = model_dir
        
        self.model.load_weights(model_dir)
        print("Loaded weights from: " + model_dir)
       
    def predict_images(self, image_dir):
        """Perform prediction on images found in ``image_dir``
        
        Parameters
        ----------
        image_dir : `str`
            Directory containing the images to perform prediction on
            
        Returns
        ----------
        image : `array_like`
            Last image that prediction was perfromed on
        """
        # load image list
        image_list = self.list_images(image_dir)
        
        for image_path in image_list:
            image = self.load_image(image_path = image_path)
            
            # percentile normalization
            if self.config.get_parameter("percentile_normalization"):
                image, _, _ = self.percentile_normalization(image, in_bound = self.config.get_parameter("percentile"))
            
            if self.config.get_parameter("tile_overlap_size") == [0,0]:
                padding = None
                if image.shape[0] < self.config.get_parameter("tile_size")[0] or image.shape[1] < self.config.get_parameter("tile_size")[1]:
                    image, padding = self.pad_image(image, image_size = self.config.get_parameter("tile_size"))
                input_image = image[np.newaxis,:,:,np.newaxis]
                
                output_image = self.model.predict(input_image, verbose=1)
                
                if padding is not None: 
                    h, w = output_image.shape[1:3]
                    output_image = np.reshape(output_image, (h, w))
                    output_image = self.remove_pad_image(output_image, padding = padding)
            else:
                tile_image_list, num_rows, num_cols, padding = self.tile_image(image, self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"))
                
                pred_train_list = []
                for tile in tile_image_list:

                    # reshape image to correct dimensions for unet
                    h, w = tile.shape[:2]
                    
                    tile = np.reshape(tile, (1, h, w, 1))

                    pred_train_list.extend(self.model.predict(tile, verbose=1))

                output_image = self.untile_image(pred_train_list, self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"),
                                                 num_rows, num_cols, padding = padding)
            
            self.save_image(output_image, image_path)
            
        return output_image
    
    def save_image(self, image, image_path, subfolder = 'Masks', suffix = '-preds'):
        """Saves image to image_path
        
        Final location of image is as follows:
          - image_path
              - subfolder
                 - model/weights file name
        
        Parameters
        ----------
        image : `array_like`
            Image to be saved
        image_path : `str`
            Location to save the image in
        subfolder : `str`
            [Default: 'Masks'] Subfolder in which the image is to be saved in
        suffix : `str`
            [Default: '-preds'] Suffix to append to the filename of the predicted image
        """
        image_dir = os.path.dirname(image_path)
        
        output_dir = os.path.join(image_dir, subfolder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        basename, _ = os.path.splitext(os.path.basename(self.weights_path))
        
        output_dir = os.path.join(output_dir, basename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename, _ = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(output_dir, "{}{}.tif".format(filename, suffix))
        
        skimage.io.imsave(output_path, image)