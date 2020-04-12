import configparser

import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

from generator import Generator
from pre_process import pre_processing
from utils import visualize, group_images


def load_model(name, best_last):

    model = model_from_json(
        open('./logs/' + name + '_architecture.json').read())
    model.load_weights('./logs/' + name + '_' + best_last + '_weights.h5')
    model.summary()

    return model


def merge(outputs, masks):
    """ merge sub_outputs to full.
    """
    assert (len(outputs.shape) == 4)
    assert (len(masks.shape) == 4)

    height = masks.shape[1]
    width = masks.shape[2]

    h_num = 1 + int((height - 48) / 16)
    w_num = 1 + int((width - 48) / 16)
    pad_height = h_num * 16 + 48
    pad_width = w_num * 16 + 48
    
    sub_num = (h_num + 1) * (w_num + 1)
    full_num = masks.shape[0]
    
    offset = 0
    merge_outputs = []

    for i in range(full_num):

        full = np.zeros((pad_height, pad_width, 1))
        
        # 4 trig
        full[:16, :16] = outputs[offset][:16, :16]
        full[:16, -16:] = outputs[offset + w_num][:16, -16:]
        full[-16:, :16] = outputs[offset + (w_num + 1) * h_num][-16:, :16]
        full[-16:, -16:] = outputs[offset + sub_num - 1][-16:, -16:]
        
        # 4 side
        for h in range(h_num + 1):
            full[16 * (h + 1):16 * (h + 2), :16] = outputs[
                offset + h * (w_num + 1)][16:32, :16]
            full[16 * (h + 1):16 * (h + 2), -16:] = outputs[
                offset + (h + 1) * (w_num + 1) - 1][16:32, -16:]
        
        for w in range(w_num + 1):
            full[:16, 16 * (w + 1):16 * (w + 2)] = outputs[
                offset + w][:16, 16:32]
            full[-16:, 16 * (w + 1):16 * (w + 2)] = outputs[
                offset + h_num * (w_num + 1) + w][-16:, 16:32]
            
        # inside
        for h in range(h_num + 1):
            for w in range(w_num + 1):
                full[16 * (h + 1):16 * (h + 2), 16 * (w + 1):16 * (w + 2)] = (
                    outputs[offset + h * (w_num + 1) + w][16:32, 16:32])

        merge_outputs.append(full)
        offset = (i + 1) * sub_num

    merge_outputs = np.array(merge_outputs).astype(np.uint8)
    merge_outputs = merge_outputs[:, :height, :width]
    merge_outputs = merge_outputs * masks
    
    return merge_outputs


def main():

    print('--------Load config--------')
    config = configparser.RawConfigParser()
    config.read('config.txt')
    
    name = config.get('evaluate', 'name')
    best_last = config.get('evaluate', 'best_last')
    datasets = config.get('evaluate', 'datasets')
    print('model: {}'.format(name))
    print('weights: {}'.format(best_last))
    print('datasets: {}'.format(datasets))

    sub_images, images, labels, masks = Generator(datasets, 'test', config)()
    
    """
    print(sub_images.shape, sub_images.dtype,
          np.max(sub_images), np.min(sub_images))
    print(labels.shape, labels.dtype, np.max(labels), np.min(labels))
    print(masks.shape, masks.dtype, np.max(masks), np.min(masks))
    """
    
    print('--------Load model--------')
    unet = load_model(name, best_last)
    outputs = unet.predict(sub_images[:35*34])
    outputs = np.argmax(outputs, axis=3)

    outputs = merge(outputs.reshape([35*34, 48, 48, 1]), masks[:1])

    print(outputs[0].shape, images[0].shape, labels[0].shape)

    outputs = np.concatenate((
        np.concatenate((images[0], labels[0]), axis=2), outputs[0]), axis=2)
    visualize(outputs, './logs/retina_pred.png')

if __name__ == '__main__':

    main()