""" Extract a subimage to test model.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

from pre_process import pre_processing
from utils import visualize

unet = model_from_json(open('./logs/retina_architecture.json').read())
unet.load_weights('./logs/retina_best_weights.h5')
unet.summary()

images = np.asarray(Image.open('./datasets/DRIVE/test/images/01_test.tif')).reshape([1, 584, 565, 3])
label = np.asarray(Image.open('./datasets/DRIVE/test/1st_manual/01_manual1.gif')) / 255.
images = pre_processing(images)

sub_images = images[0][200:248, 200:248].reshape([1, 48, 48, 1])
sub_label = label[200:248, 200:248].reshape([48, 48, 1])

pred = unet.predict(sub_images)[0]
logits = np.argmax(pred, axis=2).astype(np.uint8)

print('logits: ')
print(np.sum(logits))
print(logits.shape, logits.dtype)

outputs = np.concatenate((sub_images[0], logits.reshape((48, 48, 1))), axis=0)
outputs = np.concatenate((outputs, sub_label), axis=0)
visualize(outputs, './logs/retina_pred.png')
