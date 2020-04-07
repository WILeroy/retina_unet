import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers, optimizers

def unet(inputs_shape):
    
    def _unet():
        
        inputs = layers.Input(shape=inputs_shape, name='inputs')

        conv1_1 = layers.Conv2D(64, 3, padding='SAME', activation='relu',
                                name='conv1_1')(inputs)
        conv1_2 = layers.Conv2D(64, 3, padding='SAME', activation='relu',
                                name='conv1_2')(conv1_1)
        maxpool1 = layers.MaxPool2D(name='maxpool1')(conv1_2)

        conv2_1 = layers.Conv2D(128, 3, padding='SAME', activation='relu',
                                name='conv2_1')(maxpool1)
        conv2_2 = layers.Conv2D(128, 3, padding='SAME', activation='relu',
                                name='conv2_2')(conv2_1)
        maxpool2 = layers.MaxPool2D(name='maxpool2')(conv2_2)

        conv3_1 = layers.Conv2D(256, 3, padding='SAME', activation='relu',
                                name='conv3_1')(maxpool2)
        conv3_2 = layers.Conv2D(256, 3, padding='SAME', activation='relu',
                                name='conv3_2')(conv3_1)
        maxpool3 = layers.MaxPool2D(name='maxpool3')(conv3_2)

        conv4_1 = layers.Conv2D(512, 3, padding='SAME', activation='relu',
                                name='conv4_1')(maxpool3)
        conv4_2 = layers.Conv2D(512, 3, padding='SAME', activation='relu',
                                name='conv4_2')(conv4_1)
        maxpool4 = layers.MaxPool2D(name='maxpool4')(conv4_2)

        conv5_1 = layers.Conv2D(1024, 3, padding='SAME', activation='relu',
                                name='conv5_1')(maxpool4)
        conv5_2 = layers.Conv2D(1024, 3, padding='SAME', activation='relu',
                                name='conv5_2')(conv5_1)

        upsamping1 = layers.UpSampling2D(name='upsampling1')(conv5_2)
        upconv1 = layers.Conv2D(512, 2, padding='SAME', activation='relu',
                                name='upconv1')(upsamping1)
        concat1 = layers.concatenate([conv4_2, upconv1], axis=3, name='concat1')
        conv6_1 = layers.Conv2D(512, 3, padding='SAME', activation='relu',
                                name='conv6_1')(concat1)
        conv6_2 = layers.Conv2D(512, 3, padding='SAME', activation='relu',
                                name='conv6_2')(conv6_1)

        upsamping2 = layers.UpSampling2D(name='upsampling2')(conv6_2)
        upconv2 = layers.Conv2D(256, 2, padding='SAME', activation='relu',
                                name='upconv2')(upsamping2)
        concat2 = layers.concatenate([conv3_2, upconv2], axis=3, name='concat2')
        conv7_1 = layers.Conv2D(256, 3, padding='SAME', activation='relu',
                                name='conv7_1')(concat2)
        conv7_2 = layers.Conv2D(256, 3, padding='SAME', activation='relu',
                                name='conv7_2')(conv7_1)
    
        upsamping3 = layers.UpSampling2D(name='upsampling3')(conv7_2)
        upconv3 = layers.Conv2D(128, 2, padding='SAME', activation='relu',
                                name='upconv3')(upsamping3)
        concat3 = layers.concatenate([conv2_2, upconv3], axis=3, name='concat3')
        conv8_1 = layers.Conv2D(128, 3, padding='SAME', activation='relu',
                                name='conv8_1')(concat3)
        conv8_2 = layers.Conv2D(128, 3, padding='SAME', activation='relu',
                                name='conv8_2')(conv8_1)

        upsamping4 = layers.UpSampling2D(name='upsampling4')(conv8_2)
        upconv4 = layers.Conv2D(64, 2, padding='SAME', activation='relu',
                                name='upconv4')(upsamping4)
        concat4 = layers.concatenate([conv1_2, upconv4], axis=3, name='concat4')
        conv9_1 = layers.Conv2D(64, 3, padding='SAME', activation='relu',
                                name='conv9_1')(concat4)
        conv9_2 = layers.Conv2D(64, 3, padding='SAME', activation='relu',
                                name='conv9_2')(conv9_1)

        outputs = layers.Conv2D(1, 1, activation='sigmoid',
                                name='outputs')(conv9_2)

        return Model(inputs=inputs, outputs=outputs)
        
    return _unet()

if __name__ == '__main__':
    mynet = unet([512, 512, 3])
    mynet.summary()