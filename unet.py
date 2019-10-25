import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, optimizers
import mydataset

class unet(Model):
    def __init__(self):
        super(unet, self).__init__()

        self.conv1_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv1_1')
        self.conv1_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv1_2')
        self.maxpool1 = layers.MaxPool2D(name='maxpool1')

        self.conv2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2_1')
        self.conv2_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv2_2')
        self.maxpool2 = layers.MaxPool2D(name='maxpool2')

        self.conv3_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv3_1')
        self.conv3_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv3_2')
        self.maxpool3 = layers.MaxPool2D(name='maxpool3')

        self.conv4_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv4_1')
        self.conv4_2 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv4_2')
        self.maxpool4 = layers.MaxPool2D(name='maxpool4')

        self.conv5_1 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv5_1')
        self.conv5_2 = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv5_2')

        self.upsamping1 = layers.UpSampling2D(name='upsampling1')
        self.upconv1 = layers.Conv2D(filters=512, kernel_size=(2, 2), padding='SAME', activation='relu', name='upconv1')
        self.conv6_1 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv6_1')
        self.conv6_2 = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv6_2')

        self.upsamping2 = layers.UpSampling2D(name='upsampling2')
        self.upconv2 = layers.Conv2D(filters=256, kernel_size=(2, 2), padding='SAME', activation='relu', name='upconv2')
        self.conv7_1 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv7_1')
        self.conv7_2 = layers.Conv2D(filters=256, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv7_2')

        self.upsamping3 = layers.UpSampling2D(name='upsampling3')
        self.upconv3 = layers.Conv2D(filters=128, kernel_size=(2, 2), padding='SAME', activation='relu', name='upconv3')
        self.conv8_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv8_1')
        self.conv8_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv8_2')

        self.upsamping4 = layers.UpSampling2D(name='upsampling4')
        self.upconv4 = layers.Conv2D(filters=64, kernel_size=(2, 2), padding='SAME', activation='relu', name='upconv4')
        self.conv9_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv9_1')
        self.conv9_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='SAME', activation='relu', name='conv9_2')
        self.conv9_3 = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='relu', name='conv9_3')

    def call(self, inputs):
        conv1 = self.conv1_1(inputs)
        conv1 = self.conv1_2(conv1)
        pool1 = self.maxpool1(conv1)

        conv2 = self.conv2_1(pool1)
        conv2 = self.conv2_2(conv2)
        pool2 = self.maxpool2(conv2)

        conv3 = self.conv3_1(pool2)
        conv3 = self.conv3_2(conv3)
        pool3 = self.maxpool3(conv3)

        conv4 = self.conv4_1(pool3)
        conv4 = self.conv4_2(conv4)
        pool4 = self.maxpool4(conv4)

        conv5 = self.conv5_1(pool4)
        conv5 = self.conv5_2(conv5)

        concat1 = layers.concatenate([conv4, self.upconv1(self.upsamping1(conv5))], axis=3, name='concat1')
        conv6 = self.conv6_1(concat1)
        conv6 = self.conv6_2(conv6)

        concat2 = layers.concatenate([conv3, self.upconv2(self.upsamping2(conv6))], axis=3, name='concat2')
        conv7 = self.conv7_1(concat2)
        conv7 = self.conv7_2(conv7)

        concat3 = layers.concatenate([conv2, self.upconv3(self.upsamping3(conv7))], axis=3, name='concat3')
        conv8 = self.conv8_1(concat3)
        conv8 = self.conv8_2(conv8)

        concat4 = layers.concatenate([conv1, self.upconv4(self.upsamping4(conv8))], axis=3, name='concat4')
        conv9 = self.conv9_1(concat4)
        conv9 = self.conv9_2(conv9)
        outputs = self.conv9_3(conv9)

        return outputs

train_folder = './data/myvoc/train'
train_list_file_path = './data/myvoc/trainlist.txt'
dataset = mydataset.mydataset(train_folder, train_list_file_path)
images, labels = dataset.load()
labels.reshape([20, 512, 512, 1])
mynet = unet()
mynet.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
mynet.fit(images, labels, batch_size=4, epochs=50)