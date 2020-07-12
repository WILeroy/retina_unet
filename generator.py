import configparser

import numpy as np
from tensorflow.keras.utils import to_categorical

from pre_process import pre_processing
from utils import group_images, load_hdf5, visualize
from cv2 import cv2

class Generator:
    def __init__(
        self,
        datasets_path,
        train_test,  
        height,
        width,
        pad_height,
        pad_width
    ):
        assert (train_test == 'train' or train_test == 'test')
        
        self.datasets_path = datasets_path
        self.train_test = train_test
        self.height = height
        self.width = width
        self.pad_height = pad_height
        self.pad_width = pad_width

        self.images_path = self.datasets_path + train_test + '_images.hdf5'
        self.labels_path = self.datasets_path + train_test + '_labels.hdf5'
        self.mask_path = self.datasets_path + train_test + '_masks.hdf5'

    def __call__(self):
        images = load_hdf5(self.images_path)
        labels = load_hdf5(self.labels_path)
        masks = load_hdf5(self.mask_path)

        assert(images.shape[1]==self.height and images.shape[2]==self.width)
        assert(labels.shape[1]==self.height and labels.shape[2]==self.width)
        assert(masks.shape[1]==self.height and masks.shape[2]==self.width)

        images = pre_processing(images)
        if np.max(labels) > 1:
            labels = labels / 255.
        masks = masks / 255.

        images, labels, masks = self.padding(images, labels, masks)

        print('images:', images.shape, images.dtype, np.min(images), np.max(images))
        print('labels:', labels.shape, labels.dtype, np.min(labels), np.max(labels))
        print('masks:', masks.shape, masks.dtype, np.min(masks), np.max(masks))

        return images, labels, masks
    
    def padding(self, images, labels, masks):
        num = images.shape[0]
        channels = images.shape[3]
        images_pad_shape = (num, self.pad_height, self.pad_width, channels)
        labels_pad_shape = (num, self.pad_height, self.pad_width, 1)

        pad_images = np.zeros(images_pad_shape, dtype=np.float32)
        pad_labels = np.zeros(labels_pad_shape, dtype=np.float32)
        pad_masks = np.zeros(labels_pad_shape, dtype=np.float32)

        pad_images[:, :self.height, :self.width, :] = images
        pad_labels[:, :self.height, :self.width, :] = labels
        pad_masks[:, :self.height, :self.width, :] = masks

        pad_images *= pad_masks
        pad_labels *= pad_masks

        return pad_images, pad_labels, pad_masks