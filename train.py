import configparser
import os

import numpy as np
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical

from generator import Generator
from unet import Unet
from utils import group_images, visualize

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load config.
def main():
    config = configparser.RawConfigParser()
    config.read('config.txt')

    experiment_name = config.get('train', 'name')
    if not os.path.exists('./logs/'+experiment_name):
        os.system('mkdir ./logs/'+experiment_name)
    epochs_num = int(config.get('train', 'epochs_num'))
    batch_size = int(config.get('train', 'batch_size'))

    # Load datasets.
    datasets = config.get('train', 'datasets')
    datasets_path = config.get(datasets, 'h5py_save_path')
    height = int(config.get(datasets, 'height'))
    width = int(config.get(datasets, 'width'))
    pad_height = int(config.get(datasets, 'pad_height'))
    pad_width = int(config.get(datasets, 'pad_width'))

    x_train, y_train, masks = Generator(
        datasets_path, 'train', height, width, pad_height, pad_width)()
    visualize(
        group_images(x_train, 4),
        './logs/'+experiment_name+'/train_images.png').show()
    visualize(
        group_images(y_train, 4),
        './logs/'+experiment_name+'/train_labels.png').show()
    visualize(
        group_images(masks, 4),
        './logs/'+experiment_name+'/train_masks.png').show()
    y_train = to_categorical(y_train)

    # Build model and save.
    unet = Unet((pad_height, pad_width, 1), 5)
    unet.summary()
    unet_json = unet.to_json()
    open('./logs/'+experiment_name+'/architecture.json', 'w').write(unet_json)
    plot_model(unet, to_file = './logs/'+experiment_name+'/model.png')

    # Training.
    checkpointer = ModelCheckpoint(
        filepath = './logs/'+experiment_name+'/weights.h5',
        verbose = 1,
        monitor = 'val_loss',
        mode = 'auto',
        save_best_only = True)

    unet.fit(
        x_train, y_train,
        epochs = epochs_num,
        batch_size = batch_size,
        verbose = 1,
        shuffle = True,
        validation_split = 0.1,
        #class_weight=(0.5, 1.3),
        callbacks = [checkpointer])

if __name__ == '__main__':
    main()
