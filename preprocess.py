""" Functions about image processing.
"""

import os
import shutil
from absl import app, flags
import numpy as np
from cv2 import cv2


FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', None, 'the dir of CHASEDB and DRIVE.')

def pre_processing(image_path):
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # remove normalization.

  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  image = clahe.apply(np.array(image, dtype = np.uint8))

  # remove gamma transform.

  return image


def main(argv):
  image_dirs = ['CHASEDB/training/images',
                'CHASEDB/test/images', 
                'DRIVE/test/images',
                'DRIVE/training/images']
    
  for image_dir in image_dirs:
    image_dir = os.path.join(FLAGS.data_dir, image_dir)
    new_dir = image_dir + '_pre'
    if not os.path.exists(new_dir): os.mkdir(new_dir)

    image_list = os.listdir(image_dir)
    for image_name in image_list:
      image_path = os.path.join(image_dir, image_name)
      new_path = os.path.join(new_dir, image_name)
      print('{} -> {}'.format(image_path, new_path))
      image = pre_processing(image_path)
      cv2.imwrite(new_path, image)

if __name__ == '__main__':
  app.run(main)