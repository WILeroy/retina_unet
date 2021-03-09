import functools
import os

import cv2
import numpy as np
import tensorflow as tf


def _ParseLabelfile(file_path, preprocess):
  """ Parse label file to data list.

  Args:
    file_path: str, file path of label file.
    preprocess: bool, true if to use the preprocessed image.

  Returns:
    3 lists, paths of image, label and mask.
  """
  image_paths = []
  label_paths = []
  mask_paths = []
  
  data_dir = os.path.dirname(file_path)

  with open(file_path, 'r') as f:
    image_dir, label_dir, mask_dir = f.readline().strip().split(',')
    
    # To use the path of preprocessed image.
    if preprocess: image_dir = image_dir + '_pre'
    
    for line in f.readlines():
      image_name, label_name, mask_name = line.strip().split(',')
      image_paths.append(os.path.join(data_dir, image_dir, image_name))
      label_paths.append(os.path.join(data_dir, label_dir, label_name))
      mask_paths.append(os.path.join(data_dir, mask_dir, mask_name))
  
  return image_paths, label_paths, mask_paths


def _ParseFunction(
  image_path, label_path, mask_path, image_size, augmentation):
  """ Parse image path to image, and augmentation.

  Args:
    image_path, label_path, mask_path: Tensor (str), the path of data.
    image_size: int, the image size for the decode image (label, mask), on
      each side.
    augmentation: bool, true if the image will be augmented.

  Returns:
    image: Tensor, the processed image.
    label: Tensor, the ground-truth label.
    mask: Tensor, the RoI of image and label.
  """
  def _load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image)
    # if DRIVE:
    # image = tf.squeeze(image)
    image = tf.image.resize_with_crop_or_pad(
      image, image_size, image_size)
    return tf.cast(image, tf.float32)

  def _augmentation(image, label, mask):
    # To-Do: random transform, such as flip, rotate, etc.
    pass

  image = _load_image(image_path)
  label = tf.clip_by_value(_load_image(label_path), 0, 1)
  mask = tf.clip_by_value(_load_image(mask_path), 0, 1)

  return image, label, mask


def CreateDataset(file_path,
                  image_size,
                  batch_size,
                  preprocess=False,
                  augmentation=False,
                  seed=0,
                  repeat=True):
  """ Creates a dataset.

  Args:
    file_path: str, file path of label file.
    image_size: int, image size.
    batch_size: int, batch size.
    preprocess: bool, true if to use the preprocessed image.
    augmentation: bool, whether to apply augmentation.
    seed: int, seed for shuffling the dataset.
    repeat: bool, False if to create a test dataset.

  Returns:
    tf.data.Dataset.
  Raises:
    ValueError: if the label file does not exists.
  """

  if not os.path.exists(file_path):
    raise ValueError('label file {} does not exists.'.format(file_path))

  data_list = _ParseLabelfile(file_path, preprocess)
  dataset = tf.data.Dataset.from_tensor_slices(data_list)
  if repeat: dataset = dataset.repeat().shuffle(buffer_size=20, seed=seed)

  parse_func = functools.partial(
    _ParseFunction,
    image_size=image_size,
    augmentation=augmentation)

  dataset = dataset.map(parse_func)
  dataset = dataset.batch(batch_size)
  
  return dataset