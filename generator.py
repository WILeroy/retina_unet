import configparser

import numpy as np
from tensorflow.keras.utils import to_categorical

from pre_process import pre_processing
from utils import group_images, load_hdf5, visualize
from cv2 import cv2

class Generator:

    def __init__(self, name, config):
        """
        name: CHASEDB or DRIVE.
        """

        self.name = name
        self.height = int(config.get(name, 'height'))
        self.width = int(config.get(name, 'width'))
        self.sub_height = int(config.get('generator', 'sub_height'))
        self.sub_width = int(config.get('generator', 'sub_width'))
        self.stride_h = int(config.get('generator', 'stride_h'))
        self.stride_w = int(config.get('generator', 'stride_w'))
        self.datasets_path = config.get(name, 'h5py_save_path')
        self.images_path = self.datasets_path + 'train_images.hdf5'
        self.labels_path = self.datasets_path + 'train_labels.hdf5'
        self.mask_path = self.datasets_path + 'train_masks.hdf5'


    def __call__(self):

        images = load_hdf5(self.images_path)
        labels = load_hdf5(self.labels_path)

        images = pre_processing(images)
        labels = labels / 255.

        assert (np.min(images) >= 0 and np.max(images) <= 1)
        assert (np.min(labels) == 0 and np.max(labels) == 1)

        visualize(group_images(images, 5),
              save_path = './logs/' + self.name + '_images_pre.png')#.show()
        visualize(group_images(labels, 5),
              save_path = './logs/' + self.name + '_labels_pre.png')#.show()

        return self.extract_ordered(images, labels)


    def extract_ordered(self, images, labels):
        print('--------Extract subimages--------')
        h_num = 1 + int((self.height - self.sub_height) / self.stride_h)
        w_num = 1 + int((self.width - self.sub_width) / self.stride_w)

        print('images height: {}, width: {}'.format(self.height, self.width))
        print('subimages height: {}, width: {}'.format(self.sub_height,
                                                       self.sub_width))
        print('extract stride: ({}, {})'.format(self.stride_h, self.stride_w))

        num_of_extract = images.shape[0] * (h_num + 1) * (w_num + 1)
        sub_images = np.empty(
            (num_of_extract, self.sub_height, self.sub_width, 1),
            dtype=np.float32)
        sub_labels = np.empty(
            (num_of_extract, self.sub_height, self.sub_width, 1),
            dtype=np.float32)

        pad_height = h_num * self.stride_h + self.sub_height
        pad_width = w_num * self.stride_w + self.sub_width
        pad_image = np.zeros((pad_height, pad_width, 1), dtype=np.float32)
        pad_label = np.zeros((pad_height, pad_width, 1), dtype=np.float32)

        print('pad images to : ({}, {})'.format(pad_height, pad_width))
        print('subimages num of per col: {}'.format(h_num+1))
        print('subimages num of per row: {}'.format(w_num+1))

        count = 0
        for i in range(images.shape[0]):
            pad_image[:self.height, :self.width] = images[i]
            pad_label[:self.height, :self.width] = labels[i]

            for h in range(h_num+1):
                hstart = h * self.stride_h
                hend = hstart + self.sub_height
                sub_images[count] = pad_image[hstart:hend, :self.sub_width]
                sub_labels[count] = pad_label[hstart:hend, :self.sub_width]
                count += 1
                for w in range(w_num):
                    wstart = (w + 1) * self.stride_w
                    wend = wstart + self.sub_width
                    sub_images[count] = pad_image[hstart:hend, wstart:wend]
                    sub_labels[count] = pad_label[hstart:hend, wstart:wend]
                    count += 1

        assert (count == num_of_extract)

        return sub_images, sub_labels

def main():
    config = configparser.RawConfigParser()
    config.read('config.txt')

    sub_images, sub_labels = Generator('DRIVE', config)()
    # onehot
    one_hot = to_categorical(sub_labels)

    #print(sub_images.shape, sub_images.dtype, np.max(sub_images), np.min(sub_images))
    #print(sub_labels.shape, sub_labels.dtype, np.max(sub_labels), np.min(sub_labels))
    #print(one_hot.shape, one_hot.dtype, np.max(one_hot), np.min(one_hot))

    visualize(group_images(sub_images[:34*35], 34),
          save_path = './logs/' + 'subimages.png')
    visualize(group_images(sub_labels[:34*35], 34),
          save_path = './logs/' + 'sublabels.png')
    visualize(group_images(one_hot[:34*35, :, :, :1], 34),
          save_path = './logs/' + 'onehot0.png')
    visualize(group_images(one_hot[:34*35, :, :, 1:], 34),
          save_path = './logs/' + 'onehot1.png')