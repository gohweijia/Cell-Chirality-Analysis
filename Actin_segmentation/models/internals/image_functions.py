import os
import glob
import sys

import math
import numpy as np

#TODO: change to cv2?
import skimage
import skimage.io as skio

class Image_Functions():
    def list_images(self, image_dir, image_ext = '*.tif'):
        """List images in the directory with the given file extension

        Parameters
        ----------
        image_dir : `str`
            Directory to look for image files
        image_ext : `str`, optional
            [Default: '*.tif'] File extension of the image file
            
        Returns
        ----------
        image_list : `list`
            List of images found in the directory with the given file extension
            
        Notes
        ----------
        For linux based systems, please ensure that the file extensions are either in all lowercase or all uppercase.
        """
        # to bypass case sensitivity of file extensions in linux and possibly other systems
        if sys.platform in ["win32",]:
            image_extension = [image_ext]
        else:
            image_extension = [image_ext.lower(),image_ext.upper()]
        
        image_list = []
        for ext in image_extension:
            image_list.extend(glob.glob(os.path.join(image_dir,ext)))
            
        return image_list
    
    #######################
    # Image IO functions
    #######################
    def load_image(self, image_path, subfolder = 'Images', image_index = 0, image_ext = '*.tif'):
        """Loads images found in ``image_path``

        Parameters
        ----------
        image_path : `str`
            Path to look for image files
        subfolder : `str`, optional
            [Default: 'Images'] Subfolder in which to look for the image files
        image_index : `int`, optional
            [Default: 0] Index of image to load
        image_ext : `str`, optional
            [Default: '*.tif'] File extension of the image file
            
        Returns
        ----------
        image : `array_like`
            Loaded image
            
        Notes
        ----------
        Only one image from in each directory is loaded.
        """
        if os.path.isdir(image_path) is True:
            image_list = self.list_images(os.path.join(image_path, subfolder), image_ext = image_ext)
            if len(image_list) > 1:
               warnings.warn("More that 1 image found in directory. Loading {}".format(image_list[image_index]))
            # Load image
            image = skio.imread(image_list[image_index])
        else:
            image = skio.imread(image_path)
            
        return image
        
    def load_ground_truth(self, image_path, subfolder = 'Masks', image_ext = '*.tif'):
        """Loads ground truth images found in ``image_path`` and performs erosion/dilation/inversion if needed

        Parameters
        ----------
        image_path : `str`
            Path to look for ground truth images
        subfolder : `str`, optional
            [Default: 'Masks'] Subfolder in which to look for the ground truth images
        image_ext : `str`, optional
            [Default: '*.tif'] File extension of ground truth image file

        Returns
        ----------
        output_ground_truth : `list`
            List of ground truth images found in the directory with the given file extension
            
        class_ids : `list`
            List of class ids of the ground truth images
        """
        image_list = self.list_images(os.path.join(image_path, subfolder), image_ext = image_ext)
        
        output_ground_truth = []
        class_ids = []
        
        for ground_truth_path in image_list:
            # add class if not in list
            ground_truth_name = ground_truth_path.split('\\')[-1]
            class_name = ground_truth_name.split('_')[0]
            
            # obtain class_id
            class_ids.append(self.get_class_id(class_name))
            
            # Load image
            ground_truth_img = skio.imread(ground_truth_path)
            
            # perform erosion so that the borders will still be there after augmentation
            if self.config.get_parameter("use_binary_erosion") is True:
                from skimage.morphology import binary_erosion, disk
                # sets dtype back to unsigned integer in order for some augmentations to work
                ground_truth_dtype = ground_truth_img.dtype
                ground_truth_img = binary_erosion(ground_truth_img, disk(self.config.get_parameter("disk_size")))
                ground_truth_img = ground_truth_img.astype(ground_truth_dtype)
            
            if self.config.get_parameter("use_binary_dilation") is True:
                from skimage.morphology import binary_dilation, disk
                ground_truth_dtype = ground_truth_img.dtype
                ground_truth_img = binary_dilation(ground_truth_img, disk(self.config.get_parameter("disk_size")))
                ground_truth_img = ground_truth_img.astype(ground_truth_dtype)
            
            # perform inversion of ground_truth if needed
            if self.config.get_parameter("invert_ground_truth") is True:
                ground_truth_img = skimage.util.invert(ground_truth_img)
                
            output_ground_truth.append(ground_truth_img)
            
        return output_ground_truth, class_ids
    
    def reshape_image(self, image):
        """Reshapes the image to the correct dimenstions for Unet

        Parameters
        ----------
        image : `array_like`
            Image to be reshaped

        Returns
        ----------
        image : `array_like`
            Reshaped image 
        """
        h, w = image.shape[:2]
        image = np.reshape(image, (h, w, -1))
        return image
    
    #######################
    # Image padding
    #######################
    def pad_image(self, image, image_size, mode = 'constant'):
        """Pad image to specified image_size

        Parameters
        ----------
        image : `array_like`
            Image to be padded
        image_size : `list`
            Final size of padded image
        mode : `str`, optional
            [Default: 'constant'] Mode to pad the image

        Returns
        ----------
        image : `array_like`
            Padded image
            
        padding : `list`
            List containing the number of pixels padded to each direction
        """
        h, w = image.shape[:2]
        
        top_pad = (image_size[0] - h) // 2
        bottom_pad = image_size[0] - h - top_pad
            
        left_pad = (image_size[1] - w) // 2
        right_pad = image_size[1] - w - left_pad

        padding = ((top_pad, bottom_pad), (left_pad, right_pad))
        image = np.pad(image, padding, mode = mode, constant_values=0)
        
        return image, padding
    
    def remove_pad_image(self, image, padding):
        """Removes pad from image

        Parameters
        ----------
        image : `array_like`
            Padded image
        padding : `list`
            List containing the number of padded pixels in each direction

        Returns
        ----------
        image : `array_like`
            Image without padding
        """
        
        h, w = image.shape[:2]
        
        return image[padding[0][0]:h-padding[0][1], padding[1][0]:w-padding[1][1]]
    
    #######################
    # Tiling functions
    #######################
    def tile_image(self, image, tile_size, tile_overlap_size):
        """Converts an image into a list of tiled images

        Parameters
        ----------
        image : `array_like`
            Image to be tiled
        tile_size : `list`
            Size of each individual tile
        tile_overlap_size : `list`
            Amount of overlap (in pixels) between each tile

        Returns
        ----------
        image : `array_like`
            Image without padding
        """
        image_height, image_width = image.shape[:2]
        tile_height = tile_size[0] - tile_overlap_size[0] * 2
        tile_width = tile_size[1] - tile_overlap_size[1] * 2
        
        if image_height <= tile_height and image_width <= tile_width:
            return image
        
        num_rows = math.ceil(image_height/tile_height)
        num_cols = math.ceil(image_width/tile_width)
        num_tiles = num_rows*num_cols
        
        
        # pad image to fit tile size
        image, padding = self.pad_image(image, (tile_height*num_rows + tile_overlap_size[0] * 2, tile_width*num_cols + tile_overlap_size[1]*2))
        
        tile_image_list = []
        
        for tile_no in range(num_tiles):
            tile_x_start = (tile_no // num_rows) * tile_width
            tile_x_end = tile_x_start + tile_size[1]
            
            tile_y_start = (tile_no % num_rows) * tile_height
            tile_y_end = tile_y_start + tile_size[0]
            
            tile_image = image[tile_y_start: tile_y_end, tile_x_start:tile_x_end]
            
            # ensure input into unet is of correct shape
            tile_image = self.reshape_image(tile_image)
            
            tile_image_list.append(tile_image)
            
        return tile_image_list, num_rows, num_cols, padding
    
    def untile_image(self, tile_list, tile_size, tile_overlap_size, num_rows, num_cols, padding): 
        """Stitches a list of tiled images back into a single image

        Parameters
        ----------
        tile_list : `list`
            List of tiled images
        tile_size : `list`
            Size of each individual tile
        tile_overlap_size : `list`
            Amount of overlap (in pixels) between each tile
        num_rows : `int`
            Number of rows of tiles
        num_cols : `int`
            Number of cols of tiles
        padding : `list`
            Amount of padding used during tiling

        Returns
        ----------
        image : `array_like`
            Image without padding
        """
        if num_rows == 1 and num_cols == 1:
            image = tile_list[0]
            
            image = self.remove_pad_image(image, padding = padding)
                
            return image
              
        tile_height = tile_size[0] - tile_overlap_size[0] * 2
        tile_width = tile_size[1] - tile_overlap_size[1] * 2
        
        num_tiles = num_rows*num_cols
        
        for col in range(num_cols):
            for row in range(num_rows):
                tile_image = tile_list[num_rows*col + row][:,:,0]
                tile_image = tile_image[tile_overlap_size[0]:min(-tile_overlap_size[0],-1),tile_overlap_size[1]:min(-tile_overlap_size[1],-1)]
                if row == 0:
                    image_col = np.array(tile_image)
                else:
                    image_col = np.vstack((image_col, tile_image))
            
            if col == 0:
                image = image_col
            else:
                image = np.hstack((image, image_col))
        
        image, _ = self.pad_image(image, image_size = (tile_height * num_rows + tile_overlap_size[0] * 2, tile_width * num_cols + tile_overlap_size[1]*2))
        
        if padding is not None:
            image = self.remove_pad_image(image, padding = padding)
            
        return image
    
    
    #######################
    # Image normalization
    #######################
    def percentile_normalization(self, image, in_bound=[3, 99.8]):
        """Performs percentile normalization on the image

        Parameters
        ----------
        image : `array_like`
            Image to be normalized
        in_bound : `list`
            Upper and lower percentile used to normalize image

        Returns
        ----------
        image : `array_like`
            Normalized image
            
        image_min : `int`
            Min value of ``image``
            
        image_max : `int`
            Max value of ``image``
        """
        image_min = np.percentile(image, in_bound[0])
        image_max = np.percentile(image, in_bound[1])
        image = (image - image_min)/(image_max - image_min)

        return image, image_min, image_max