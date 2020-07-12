import configparser
from os import path, system

import numpy as np
from PIL import Image
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

from generator import Generator
from metric import SegmentationMetric
from pre_process import pre_processing
from unet import Unet
from utils import group_images, load_hdf5, visualize

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def load_model(name):
    model = model_from_json(
        open('./logs/'+name+'/architecture.json').read())
    #model = Unet((960, 1024, 1), 5, train=True)
    model.load_weights('./logs/'+name+'/weights.h5')
    model.summary()
    return model

def visualize_predicts(predicts, images_path, labels, masks, height, width, name):
    """
    right: green
    error: red
    missing: blue
    """
    if not path.exists('./logs/'+name+'/predicts/'):
        system('mkdir ./logs/'+name+'/predicts/')

    images = load_hdf5(images_path+'test_images.hdf5')

    pad_height = labels.shape[1]
    pad_width = labels.shape[2]

    for i in range(len(predicts)):
        vis = np.zeros((pad_height, pad_width, 3))
        right = predicts[i] * np.squeeze(labels[i]) * np.squeeze(masks[i])
        error = predicts[i] - right
        missing = np.squeeze(labels[i]) - right
        vis[:, :, 0] = error
        vis[:, :, 1] = right
        vis[:, :, 2] = missing

        vis = np.concatenate((images[i], vis[:height, :width]*255), axis=0)
        
        visualize(vis, './logs/'+name+'/predicts/'+str(i+1)+'.png')

def main():
    # Load config.
    config = configparser.RawConfigParser()
    config.read('config.txt')
    
    name = config.get('evaluate', 'name')
    if not path.exists('./logs/'+name):
        system('mkdir ./logs/'+name)
    datasets = config.get('evaluate', 'datasets')
    
    datasets_path = config.get(datasets, 'h5py_save_path')
    height = int(config.get(datasets, 'height'))
    width = int(config.get(datasets, 'width'))
    pad_height = int(config.get(datasets, 'pad_height'))
    pad_width = int(config.get(datasets, 'pad_width'))

    # Load datasets.
    images, labels, masks = Generator(
        datasets_path, 'test', height, width, pad_height, pad_width)()
    
    visualize(
        group_images(images, 4),
        './logs/'+name+'/test_images.png').show()
    visualize(
        group_images(labels, 4),
        './logs/'+name+'/test_labels.png').show()
    visualize(
        group_images(masks, 4),
        './logs/'+name+'/test_masks.png').show()

    # Load model and predict.
    unet = load_model(name)
    predicts = []
    for i in range(images.shape[0]):
        predict = unet.predict(images[i:i+1])
        predict = np.squeeze(np.argmax(predict, axis=3))
        predicts.append(predict)

    visualize_predicts(predicts, datasets_path, labels, masks, height, width, name)

    # Evaluate.
    labels = (labels - ((masks + 1) % 2)).astype(np.int8)
    metric = SegmentationMetric(2)
    mIOU = 0
    PA = 0
    mPA = 0
    for i in range(len(predicts)):
        metric.add_batch(np.squeeze(labels[i]), predicts[i].astype(np.int8))
        mIOU += metric.mean_intersection_over_union()
        PA += metric.pixel_accuracy()
        mPA += metric.mean_pixel_accuracy()
        metric.reset()
    
    result_eva = 'mIOU: {:.6f}\nPA: {:.6f}\nmPA: {:.6f}'.format(
        mIOU/len(predicts), PA/len(predicts), mPA/len(predicts))
    print(result_eva)
    with open('./logs/'+name+'/evaluate.txt', 'w') as f:
        f.write(result_eva)

if __name__ == '__main__':
    main()
