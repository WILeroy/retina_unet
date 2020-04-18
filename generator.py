import configparser

import numpy as np
from tensorflow.keras.utils import to_categorical

from pre_process import pre_processing
from utils import group_images, load_hdf5, visualize
from cv2 import cv2

class Generator:
    def __init__(self, name, train_test, config):
        """
        name: CHASEDB or DRIVE.
        """
        self.name = name
        self.train_test = train_test
        assert (name == 'CHASEDB' or name == 'DRIVE')
        assert (train_test == 'train' or train_test == 'test')

        self.height = int(config.get(name, 'height'))
        self.width = int(config.get(name, 'width'))
        self.datasets_path = config.get(name, 'h5py_save_path')

        self.sub_height = int(config.get('generator', 'sub_height'))
        self.sub_width = int(config.get('generator', 'sub_width'))
        self.stride_h = int(config.get('generator', 'stride_h'))
        self.stride_w = int(config.get('generator', 'stride_w'))
        assert (self.sub_height % self.stride_h == 0)
        assert (self.sub_width % self.stride_w == 0)

        self.images_path = self.datasets_path + train_test + '_images.hdf5'
        self.labels_path = self.datasets_path + train_test + '_labels.hdf5'
        self.mask_path = self.datasets_path + train_test + '_masks.hdf5'

    def __call__(self):
        images = load_hdf5(self.images_path)
        labels = load_hdf5(self.labels_path)
        masks = load_hdf5(self.mask_path)

        images = pre_processing(images)
        if np.max(labels) > 1:
            labels = labels / 255.
        masks = masks / 255.

        #visualize(group_images(images, 4)).show()
        #visualize(group_images(labels, 4)).show()
        #visualize(group_images(masks, 4)).show()

        #print(images.shape, images.dtype, np.min(images), np.max(images))
        #print(labels.shape, labels.dtype, np.min(labels), np.max(labels))
        #print(masks.shape, masks.dtype, np.min(masks), np.max(masks))

        if self.train_test == 'train':
            return self.extract_ordered(images, labels)

        if self.train_test == 'test':
            sub_images, sub_labels = self.extract_ordered(images, labels)
            return (sub_images, images, labels, masks)


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
    """ Unit test.
    """
    config = configparser.RawConfigParser()
    config.read('config.txt')

    sub_images, sub_labels = Generator('DRIVE', 'train', config)()
    print(sub_images.shape, sub_images.dtype, np.min(sub_images), np.max(sub_images))
    print(sub_labels.shape, sub_labels.dtype, np.min(sub_labels), np.max(sub_labels))
    visualize(group_images(sub_images[:34*35], 34)).show()
    visualize(group_images(sub_labels[:34*35], 34)).show()

    sub_images, images, labels, masks = Generator('DRIVE', 'test', config)()
    print(sub_images.shape, sub_images.dtype, np.min(sub_images), np.max(sub_images))
    print(images.shape, images.dtype, np.min(images), np.max(images))
    print(labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print(masks.shape, masks.dtype, np.min(masks), np.max(masks))
    visualize(group_images(sub_images[:34*35], 34)).show()

    sub_images, sub_labels = Generator('CHASEDB', 'train', config)()
    print(sub_images.shape, sub_images.dtype, np.min(sub_images), np.max(sub_images))
    print(sub_labels.shape, sub_labels.dtype, np.min(sub_labels), np.max(sub_labels))
    visualize(group_images(sub_images[:59*61], 61)).show()
    visualize(group_images(sub_labels[:59*61], 61)).show()

    sub_images, images, labels, masks = Generator('CHASEDB', 'test', config)()
    print(sub_images.shape, sub_images.dtype, np.min(sub_images), np.max(sub_images))
    print(images.shape, images.dtype, np.min(images), np.max(images))
    print(labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print(masks.shape, masks.dtype, np.min(masks), np.max(masks))
    visualize(group_images(sub_images[:59*61], 61)).show()

if __name__ == '__main__':
    main()