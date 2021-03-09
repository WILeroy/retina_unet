import os

import numpy as np
import tensorflow as tf
from absl import app, flags
from skimage import io
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from core.dataset import CreateDataset
from core.unet import Unet


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', None,
                    'the dir of weights file saved by model.save_weights().')
flags.DEFINE_string('label_file_path', None,
                    'Path of testing dataset label file.')
flags.DEFINE_integer('batch_size', 1, 'Global batch size.')
flags.DEFINE_integer('image_size', 1024, 'Size of each image side to use.')
flags.DEFINE_boolean('preprocess', False, 'Whether to use preprocessed image.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_boolean('transpose_conv', False,
                     'Whether to use Conv2DTranspose as upsample layer.')


def visulize_predicts(predicts, images, labels, masks, logdir, idx):
  """ Visulizing predicts with different color:
    error -> red, right -> green, miss -> blue.

  Args:
    predicts: numpy array, shape = [batch, h, w].
    images: numpy array, shape = [batch, h, w, 1 or 3].
    labels: numpy array, shape = [batch, h, w, 1].
    masks: numpy array, shape = [batch, h, w, 1].
    logdir: str, to stores the floder of visualization results.
    idx: int, use to name result image, images[i] -> str(idx+i).jpg.
  """
  visulize_dir = os.path.join(logdir, 'visulize')
  if not os.path.exists(visulize_dir): os.mkdir(visulize_dir)

  if images.shape[-1] == 1:
    images = np.tile(images, [1, 1, 1, 3])

  for i in range(predicts.shape[0]):
    right = predicts[i][:, :, np.newaxis] * labels[i] * masks[i]
    error = predicts[i][:, :, np.newaxis] - right
    miss = labels[i] - right
    vis = np.concatenate([error, right, miss], axis=-1)
    vis = np.concatenate([images[i], vis * 255], axis=0)
    vis_path = os.path.join(visulize_dir, str(idx+i)+'.png')
    io.imsave(vis_path, vis.astype(np.uint8))

  
def main(argv):
  # Load datasets.
  test_dataset = CreateDataset(
    file_path=FLAGS.label_file_path,
    image_size=FLAGS.image_size,
    batch_size=FLAGS.batch_size,
    preprocess=FLAGS.preprocess,
    augmentation=False,
    repeat=False)
  
  model = Unet(
      data_format='channels_last',
      classes=FLAGS.num_classes,
      transpose_conv=FLAGS.transpose_conv)
  weights_path = os.path.join(FLAGS.logdir, 'unet_weights')
  model.load_weights(weights_path)

  # Setup metric.
  mIoU = tf.keras.metrics.MeanIoU(num_classes=2)

  count = 0
  for idx, (images, labels, masks) in enumerate(test_dataset):
    logits = model.build_call(images, training=False)
    predicts = np.argmax(logits.numpy(), axis=-1)

    # optinal.
    #visulize_predicts(predicts, images, labels, masks, FLAGS.logdir, count)
    count += predicts.shape[0]

    mIoU.update_state(labels, predicts, masks)

  cm = mIoU.total_cm
  print('confuse mat:\n', cm.numpy())
  print('IoU: ', (cm[1, 1] / (cm[0, 1] + cm[1, 0] + cm[1, 1])).numpy())
  print('mIoU: ', mIoU.result().numpy())


if __name__ == '__main__':
    app.run(main)
