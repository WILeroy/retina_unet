import configparser
import os

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical

from generator import Generator
from unet import Unet

# Load config.
config = configparser.RawConfigParser()
config.read('config.txt')

experiment_name = config.get('train', 'name')
epochs_num = int(config.get('train', 'epochs_num'))
batch_size = int(config.get('train', 'batch_size'))

# Load datasets.
datasets = config.get('train', 'datasets')
sub_height = int(config.get('generator', 'sub_height'))
sub_width = int(config.get('generator', 'sub_width'))
x_train, y_train = Generator(datasets, 'train', config)()
y_train = to_categorical(y_train)

print(np.max(x_train), np.min(x_train), x_train.shape, x_train.dtype)
print(np.max(y_train), np.min(y_train), y_train.shape, y_train.dtype)

# Build model and save.
unet = Unet((sub_height, sub_width, 1))
unet.summary()
unet_json = unet.to_json()
open('./logs/' + experiment_name + '_architecture.json', 'w').write(unet_json)
plot_model(unet, to_file = './logs/' + experiment_name + '_model.png')

# Training.
checkpointer = ModelCheckpoint(filepath = './logs/'
                                        + experiment_name +'_best_weights.h5',
                                verbose = 1,
                                monitor = 'val_loss',
                                mode = 'auto',
                                save_best_only = True)

unet.fit(x_train, y_train,
         epochs = epochs_num,
         batch_size = batch_size,
         verbose = 1,
         shuffle = True,
         validation_split = 0.1,
         callbacks = [checkpointer])

unet.save_weights('./logs/' + experiment_name +'_last_weights.h5',
                   overwrite=True)
                   