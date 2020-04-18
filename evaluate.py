import configparser

import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

from generator import Generator
from metric import SegmentationMetric
from pre_process import pre_processing
from utils import group_images, visualize


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

        full = np.zeros([pad_height, pad_width, 1])
        
        # 4 corner.
        full[:16, :16] = outputs[offset][:16, :16]
        full[:16, -16:] = outputs[offset + w_num][:16, -16:]
        full[-16:, :16] = outputs[offset + (w_num + 1) * h_num][-16:, :16]
        full[-16:, -16:] = outputs[offset + sub_num - 1][-16:, -16:]
        
        # 4 side.
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
            
        # inside.
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


def visualize_predicts(num, masks, labels, predicts):
    """
    red: error
    green: missed
    light blue: right
    """
    outputs = labels * 2 + predicts
    outputs = to_categorical(outputs)
    vis = np.zeros([labels.shape[0], labels.shape[1], labels.shape[2], 3])
    vis[:, :, :, 0] = outputs[:, :, :, 1]
    vis[:, :, :, 1] = outputs[:, :, :, 2] + outputs[:, :, :, 3]
    vis[:, :, :, 2] = outputs[:, :, :, 3] 
    for i in range(20):
        visualize(vis[i], './logs/'+str(i+1)+'.png')


def main():
    # Load config.
    config = configparser.RawConfigParser()
    config.read('config.txt')
    
    name = config.get('evaluate', 'name')
    best_last = config.get('evaluate', 'best_last')
    
    # Load datasets.
    datasets = config.get('evaluate', 'datasets')
    sub_images, images, labels, masks = Generator(datasets, 'test', config)()
    
    print(sub_images.shape, sub_images.dtype,
          np.min(sub_images), np.max(sub_images))
    print(labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print(masks.shape, masks.dtype, np.min(masks), np.max(masks))
    
    # Load model and predict.
    unet = load_model(name, best_last)
    sub_predicts = unet.predict(sub_images)
    sub_predicts = np.argmax(sub_predicts, axis=3)
    predicts = merge(sub_predicts.reshape([-1, 48, 48, 1]), masks)

    print(predicts.shape, images.shape, labels.shape)
    visualize_predicts(20, masks, labels, predicts)

    # Evaluate.
    labels = (labels - ((masks + 1) % 2)).astype(np.int8)
    metric = SegmentationMetric(2)
    mIOU = 0
    PA = 0
    mPA = 0
    for i in range(20):
        metric.add_batch(labels[i], predicts[i].astype(np.int8))
        mIOU += metric.mean_intersection_over_union()
        PA += metric.pixel_accuracy()
        mPA += metric.mean_pixel_accuracy()
        metric.reset()
    print('mIOU:', mIOU/20)
    print('PA:', PA/20)
    print('mPA:', mPA/20)
    

if __name__ == '__main__':
    main()
