import os, sys
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from .image_functions import Image_Functions      

class Dataset(Image_Functions):
    def __init__(self):
        """Creates Dataset object that is used to manipulate the training data.
    
        Attributes
        ----------
        classes : list
            List of dictionaries containing the class name and id
            
        train_images : list
            List of images that is used as the input for the network
            
        train_ground_truth : list
            List of images that is used as the ground truth for the network
        """
            
        self.classes = []
        self.train_images = []
        self.train_ground_truth = []
        
        super().__init__()
    
    #######################
    # Class id functions
    #######################
    def get_class_id(self, class_name):
        """Returns the class id and adds class to list if not in list of classes.
    
        Parameters
        ----------
        class_name : str
            Identity of class that will be associated with the class id
            
        Returns
        ----------
        int
            Class id
        """
        
        if len(self.classes) == 0:
            self.classes.append({"class": class_name, "id": 0})
            return 0
        
        for class_info in self.classes:
            # if class exist, return class id
            if class_info["class"] == class_name:
                return class_info["id"]
   
        self.classes.append({"class": class_name, "id": len(self.classes)-1})
        return len(self.classes)-1
    
    #######################
    # Class id functions
    #######################
    def sanity_check(self, image_index):
        """Plots the augmented image and ground_truth to check if everything is ok.
    
        Parameters
        ----------
        image_index : int
            Index of the image and its corresponding ground_truth
        """
        
        image = self.aug_images[image_index][:,:,0]
        ground_truth = self.aug_ground_truth[image_index][:,:,0]

        plt.figure(figsize=(14, 14))
        plt.axis('off')
        plt.imshow(image, cmap='gray', 
                   norm=None, interpolation=None)
        plt.show()

        plt.figure(figsize=(14, 14))
        plt.axis('off')
        plt.imshow(ground_truth, cmap='gray', 
                   norm=None, interpolation=None)
        plt.show()
    
    def load_dataset(self, dataset_dir = None, tiled = False):
        """Loads dataset from ``dataset_dir``
    
        Parameters
        ----------
        dataset_dir : str or none, optional
            Folder to load the dataset from. If none, ``dataset_dir`` is obtained from config file
            
        tiled : bool, optional
            To set if tiling function is used
        """
        
        # update dataset_dir if specified. If not, load dataset_dir from config file
        if dataset_dir is None:
            dataset_dir = self.config.get_parameter("dataset_dir")
        else:
            self.config.update_parameter(self.config.find_key("dataset_dir"), dataset_dir)
        
        image_dirs = next(os.walk(dataset_dir))[1]
        image_dirs = [f for f in image_dirs if not f[0] == '.']
        
        for img_dir in image_dirs:
            # images
            image = self.load_image(os.path.join(dataset_dir, img_dir), subfolder = self.config.get_parameter("image_subfolder"))
            
            # percentile normalization
            if self.config.get_parameter("percentile_normalization"):
                image, _, _ = self.percentile_normalization(image, in_bound = self.config.get_parameter("percentile"))
            
            if tiled is True:
                tile_image_list, num_rows, num_cols, padding = self.tile_image(image, self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"))
                self.config.update_parameter(["images","num_rows"], num_rows)
                self.config.update_parameter(["images","num_cols"], num_cols)
                self.config.update_parameter(["images","padding"], padding)
                self.train_images.extend(tile_image_list)
            else:
                self.train_images.extend([image,])
            
            #ground_truth
            ground_truth, class_id = self.load_ground_truth(os.path.join(dataset_dir, img_dir), subfolder = self.config.get_parameter("ground_truth_subfolder"))
            if tiled is True:
                tile_ground_truth_list, _, _, _ = self.tile_image(ground_truth[0], self.config.get_parameter("tile_size"), self.config.get_parameter("tile_overlap_size"))
                self.train_ground_truth.extend(tile_ground_truth_list)
            else:
                self.train_ground_truth.extend(ground_truth)
                
    #######################
    # Image augmentation
    #######################
    def augment_images(self):
        """Augments images using the parameters in the config file"""
        
        # TODO: To allow for augmentation of multi-class images
        
        augmentor = self.augmentations(p=self.config.get_parameter("augmentations_p"))
        
        # increase number of images
        self.aug_images = self.train_images*self.config.get_parameter("num_augmented_images")
        self.aug_ground_truth = self.train_ground_truth*self.config.get_parameter("num_augmented_images")
        
        print("Performing augmentations on {} images".format(len(self.aug_images)))
        sys.stdout.flush()
        
        for i in tqdm(range(len(self.aug_images)),desc="Augmentation of images"):
            
            # target must be image and mask in order for albumentations to work
            data = {"image": self.aug_images[i], 
                    "mask": self.aug_ground_truth[i]}
            augmented = augmentor(**data)

            self.aug_images[i] = self.reshape_image(np.asarray(augmented["image"]))
            
            # add 
            if self.config.get_parameter("use_binary_dilation_after_augmentation") is True:
                from skimage.morphology import binary_dilation, disk
                self.aug_ground_truth[i] = self.reshape_image(binary_dilation(np.ndarray.astype(augmented["mask"], np.bool), disk(self.config.get_parameter("disk_size"))))
            else:
                self.aug_ground_truth[i] = self.reshape_image(np.ndarray.astype(augmented["mask"], np.bool))

        self.aug_images = np.stack(self.aug_images, axis = 0)
        self.aug_ground_truth = np.stack(self.aug_ground_truth, axis = 0)
        
        mean = self.aug_images.mean()
        std = self.aug_images.std()
        
        self.config.update_parameter(["images","mean"], float(mean))
        self.config.update_parameter(["images","std"], float(std))
        
        print("Augmentations complete!")

    def augmentations(self, p = None):
        """Generates list of augmentations using parameters obtained from config file
        
        Parameters
        ----------
        p : int, optional
            probability to apply any augmentations to image
        
        Returns
        ----------
        function
            function used to augment images
        """
        from albumentations import (
            RandomCrop, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
            Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, ElasticTransform,
            IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur,
            IAASharpen, RandomBrightnessContrast, Flip, OneOf, Compose
        )
        
        augmentation_list = []
        
        if self.config.get_parameter("random_rotate") is True:
            augmentation_list.append(RandomRotate90(p = self.config.get_parameter("random_rotate_p"))) # 0.9
        
        if self.config.get_parameter("flip") is True:
            augmentation_list.append(Flip())
            
        if self.config.get_parameter("transpose") is True:
            augmentation_list.append(Transpose())
            
        if self.config.get_parameter("blur_group") is True:
            blur_augmentation = []
            if self.config.get_parameter("motion_blur") is True:
                blur_augmentation.append(MotionBlur(p = self.config.get_parameter("motion_blur_p")))
            if self.config.get_parameter("median_blur") is True:
                blur_augmentation.append(MedianBlur(blur_limit = self.config.get_parameter("median_blur_limit"), p = self.config.get_parameter("median_blur_p")))
            if self.config.get_parameter("blur") is True:
                blur_augmentation.append(Blur(blur_limit = self.config.get_parameter("blur_limit"), p = self.config.get_parameter("blur_p")))
            augmentation_list.append(OneOf(blur_augmentation, p = self.config.get_parameter("blur_group_p"))) 
            
        if self.config.get_parameter("shift_scale_rotate") is True:
            augmentation_list.append(ShiftScaleRotate(shift_limit = self.config.get_parameter("shift_limit"),
                                                      scale_limit = self.config.get_parameter("scale_limit"),
                                                      rotate_limit = self.config.get_parameter("rotate_limit"),
                                                      p = self.config.get_parameter("shift_scale_rotate_p")))
        if self.config.get_parameter("distortion_group") is True:
            distortion_augmentation = []
            if self.config.get_parameter("optical_distortion") is True:
                distortion_augmentation.append(OpticalDistortion(p = self.config.get_parameter("optical_distortion_p")))
            if self.config.get_parameter("elastic_transform") is True:
                distortion_augmentation.append(ElasticTransform(p = self.config.get_parameter("elastic_transform_p")))
            if self.config.get_parameter("grid_distortion") is True:
                distortion_augmentation.append(GridDistortion(p = self.config.get_parameter("grid_distortion_p")))
            
            augmentation_list.append(OneOf(distortion_augmentation, p = self.config.get_parameter("distortion_group_p")))
        
        if self.config.get_parameter("brightness_contrast_group") is True:
            contrast_augmentation = []
            if self.config.get_parameter("clahe") is True:
                contrast_augmentation.append(CLAHE())
            if self.config.get_parameter("sharpen") is True:
                contrast_augmentation.append(IAASharpen())
            if self.config.get_parameter("random_brightness_contrast") is True:
                contrast_augmentation.append(RandomBrightnessContrast())
           
            augmentation_list.append(OneOf(contrast_augmentation, p = self.config.get_parameter("brightness_contrast_group_p")))
            
        augmentation_list.append(RandomCrop(self.config.get_parameter("tile_size")[0], self.config.get_parameter("tile_size")[1], always_apply=True))
        
        return Compose(augmentation_list, p = p)

############################### TODO ###############################
#     def preapare_data(self):
#         """        
#         Performs augmentation if needed
#         """
        
            
#     # Create data generator
#     # Return augmented images/ground_truth arrays of batch size
#     def generator(features, labels, batch_size, seq_det):
#         # create empty arrays to contain batch of features and labels
#         batch_features = np.zeros((batch_size, features.shape[1], features.shape[2], features.shape[3]))
#         batch_labels = np.zeros((batch_size, labels.shape[1], labels.shape[2], labels.shape[3]))

#         while True:
#             # Fill arrays of batch size with augmented data taken randomly from full passed arrays
#             indexes = random.sample(range(len(features)), batch_size)
#             # Perform the exactly the same augmentation for X and y
#             random_augmented_images, random_augmented_labels = do_augmentation(seq_det, features[indexes], labels[indexes])
#             batch_features[:,:,:,:] = random_augmented_images[:,:,:,:]
#             batch_labels[:,:,:,:] = random_augmented_labels[:,:,:,:]

#             yield batch_features, batch_labels
            
    # Train augmentation
#     def do_augmentation(seq_det, X_train, y_train):
#         # Use seq_det to build augmentation.
#         # ....
#         return np.array(X_train_aug), np.array(y_train_aug)

#     seq = iaa.Sequential([
#         iaa.Fliplr(0.5), # horizontally flip
#         iaa.OneOf([
#             iaa.Noop(),
#             iaa.GaussianBlur(sigma=(0.0, 1.0)),
#             iaa.Noop(),
#             iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
#             iaa.Noop(),
#             iaa.PerspectiveTransform(scale=(0.04, 0.08)),
#             iaa.Noop(),
#             iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
#         ]),
#         # More as you want ...
#     ])
#     seq_det = seq.to_deterministic()
    
#     history = model.fit_generator(generator(X_train, y_train, BATCH_SIZE, seq_det),
#                               epochs=EPOCHS,
#                               steps_per_epoch=steps_per_epoch,
#                               validation_data=(X_valid, y_valid),
#                               verbose = 1, 
#                               callbacks = [check_point]
#                              ) 
    
    # Image augmentations
            
############################### END of TODO ###############################