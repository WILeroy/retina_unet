""" Script to load datasets (DRIVE and CHASEDB) and write to *.hdf5.
"""

import os
import configparser

import h5py
import numpy as np
from PIL import Image

from utils import load_hdf5, write_hdf5, visualize, group_images


def save_DRIVE_to_h5py(train_test, num, config):
    images_path = config.get('DRIVE', train_test + '_images_path')
    labels_path = config.get('DRIVE', train_test + '_labels_path')
    masks_path = config.get('DRIVE', train_test + '_masks_path')
    height = int(config.get('DRIVE', 'height'))
    width = int(config.get('DRIVE', 'width'))

    images = np.empty((num, height, width, 3), dtype=np.float32)
    labels = np.empty((num, height, width, 1), dtype=np.float32)
    masks = np.empty((num, height, width, 1), dtype=np.float32)

    files = os.listdir(images_path)
    for i in range(len(files)):
        # read i-th image.
        images[i] = np.asarray(Image.open(images_path + files[i]))

        # read corresponding label.
        label_name = files[i][0:2] + "_manual1.gif"
        labels[i] = np.asarray(Image.open(labels_path + label_name)).reshape(
            (height, width, 1))
        
        # read corresponding mask.
        mask_name = ""
        if train_test == "train":
            mask_name = files[i][0:2] + "_training_mask.gif"
        elif train_test == "test":
            mask_name = files[i][0:2] + "_test_mask.gif"
        masks[i] = np.asarray(Image.open(masks_path + mask_name)).reshape(
            (height, width, 1))

    """ Check data. """
    print('DRIVE', train_test)
    print('images', images.shape, images.dtype, np.min(images), np.max(images))
    print('labels', labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print('masks', masks.shape, masks.dtype, np.min(masks), np.max(masks))

    """ Visualize datasets to check integrity."""
    """
    visualize(group_images(images, 5),
              save_path = './logs/DRIVE_' + train_test + '_images.png')
    visualize(group_images(labels, 5),
              save_path = './logs/DRIVE_' + train_test + '_labels.png')
    visualize(group_images(masks, 5),
              save_path = './logs/DRIVE_' + train_test + '_masks.png')
    """
    #visualize(group_images(images, 4)).show()
    #visualize(group_images(labels, 4)).show()
    #visualize(group_images(masks, 4)).show()

    save_path = config.get('DRIVE', 'h5py_save_path')
    if os.path.exists(save_path) == False:
        os.system('mkdir {}'.format(save_path))
    write_hdf5(images, save_path + train_test + '_images' + '.hdf5')
    write_hdf5(labels, save_path + train_test + '_labels' + '.hdf5')
    write_hdf5(masks, save_path + train_test + '_masks' + '.hdf5')


def save_CHASEDB_to_h5py(train_test, num, config):
    images_path = config.get('CHASEDB', train_test + '_images_path')
    labels_path = config.get('CHASEDB', train_test + '_labels_path')
    masks_path = config.get('CHASEDB', train_test + '_masks_path')
    height = int(config.get('CHASEDB', 'height'))
    width = int(config.get('CHASEDB', 'width'))

    images = np.empty((num, height, width, 3), dtype=np.float32)
    labels = np.empty((num, height, width, 1), dtype=np.float32)
    masks = np.empty((num, height, width, 1), dtype=np.float32)

    files = os.listdir(images_path)
    for i in range(len(files)):
        # read i-th image.
        images[i] = np.asarray(Image.open(images_path + files[i]))

        # read corresponding label.
        label_name = files[i][:9] + "_1stHO.png"
        labels[i] = np.asarray(Image.open(labels_path + label_name)).reshape(
            (height, width, 1))
        
        # read corresponding mask.
        mask_name = 'mask_' + files[i][6:9] + '.png'
        masks[i] = np.asarray(Image.open(masks_path + mask_name)).reshape(
            (height, width, 1))

    """ Check data. """
    print('CHASEDB', train_test)
    print('images', images.shape, images.dtype, np.min(images), np.max(images))
    print('labels', labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print('masks', masks.shape, masks.dtype, np.min(masks), np.max(masks))

    """ Visualize datasets to check integrity."""
    """
    visualize(group_images(images, 4),
              save_path = './logs/CHASEDB_' + train_test + '_images.png')
    visualize(group_images(labels, 4),
              save_path = './logs/CHASEDB_' + train_test + '_labels.png')
    visualize(group_images(masks, 4),
              save_path = './logs/CHASEDB_' + train_test + '_masks.png')
    """
    #visualize(group_images(images, 4)).show()
    #visualize(group_images(labels, 4)).show()
    #visualize(group_images(masks, 4)).show()

    save_path = config.get('CHASEDB', 'h5py_save_path')
    if os.path.exists(save_path) == False:
        os.system('mkdir {}'.format(save_path))
    write_hdf5(images, save_path + train_test + '_images' + '.hdf5')
    write_hdf5(labels, save_path + train_test + '_labels' + '.hdf5')
    write_hdf5(masks, save_path + train_test + '_masks' + '.hdf5')


def main():
    config = configparser.RawConfigParser()
    config.read('config.txt')

    save_DRIVE_to_h5py('train', 20, config)
    save_DRIVE_to_h5py('test', 20, config)
    save_CHASEDB_to_h5py('train', 20, config)
    save_CHASEDB_to_h5py('test', 8, config)


if __name__ == '__main__':
    main()