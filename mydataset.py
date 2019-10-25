from PIL import Image
import numpy as np
from skimage import io
from skimage import transform
import os

class mydataset():
    def __init__(self, folder_path, list_file_path, img_size=512):
        """ 初始化数据集类
        参数：
            folder_path : 数据集文件夹
            list_file_path　: 数据集列表（只记录每项的名字）
            img_size : 数据集图片大小（resize时使用）
        """
        self._folder_path = folder_path
        with open(list_file_path) as list_file:
            self._data_list = list_file.readlines()
        self._img_size = img_size

        self._images = np.array([self._read(data_name.strip('\n'), target='img')\
                                            for data_name in self._data_list])
        self._labels = np.array([self._read(data_name.strip('\n'), target='label')\
                                            for data_name in self._data_list])
        self._labels_onehot = np.array([self._onehot((label+0.5).astype(np.int8))\
                                                     for label in self._labels])

        print('Init dataset: %s' % folder_path)
        print('images shape: ', self._images.shape)
        print('labels shape: ', self._labels.shape)
        print('labels_onehot shape: ', self._labels_onehot.shape)

    def _read(self, data_name, target):
        """ 读取数据：self._folder_path, data_name, target+'.png'
        """
        
        data_path = os.path.join(self._folder_path, data_name)
        data_path = os.path.join(data_path, target+'.png')
        if os.path.exists(data_path) == False:
            print('File: %s doesnt exist.' % data_path)
            exit(-1)

        data = Image.open(data_path)
        pad_data = self._transform(np.array(data), target)

        # resize过程中会差值。
        return transform.resize(pad_data, [512, 512], mode='constant')

    def _onehot(self, label):
        one = np.ones(label.shape)
        return np.stack((one-label, label), axis=2)

    def _transform(self, data, target):
        if target == 'img':
            shape = data.shape
            if shape[0] < shape[1]:
                pad = np.zeros([shape[1], shape[1], shape[2]])
                pad[0:shape[0], 0:shape[1], :] = data
                return pad
            elif shape[0] > shape[1]:
                pad = np.zeros([shape[0], shape[0], shape[2]])
                pad[0:shape[0], 0:shape[1], :] = data
                return pad
        else:
            shape = data.shape
            if shape[0] < shape[1]:
                pad = np.zeros([shape[1], shape[1]])
            elif shape[0] > shape[1]:
                pad = np.zeros([shape[0], shape[0]])
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if data[i][j] != 0:
                        pad[i][j] = 1
            return pad
    
    def load(self):
        return self._images, self._labels