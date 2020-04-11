import numpy as np
from PIL import Image
from cv2 import cv2


def pre_processing(data):
    assert (len(data.shape) == 4)
    assert (data.shape[3] == 3)

    train_imgs = rgb2gray(data)
    train_imgs = datasets_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.
    
    return train_imgs.astype(np.float32)


# convert RGB image to gray image.
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)
    assert (rgb.shape[3] == 3)
    gray = (rgb[:, :, :, 0] * 0.299 +
            rgb[:, :, :, 1] * 0.587 +
            rgb[:, :, :, 2] * 0.114)
    gray = np.reshape(gray, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return gray


def clahe_equalized(images):
    assert (len(images.shape) == 4)
    assert (images.shape[3] == 1) 

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    images_equalized = np.empty(images.shape)
    for i in range(images.shape[0]):
        images_equalized[i, :, :, 0] = clahe.apply(np.array(images[i, :, :, 0],
                                                            dtype = np.uint8))

    return images_equalized


def datasets_normalized(images):
    assert (len(images.shape) == 4)
    assert (images.shape[3] == 1)

    images_normalized = np.empty(images.shape)
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / images_std
    for i in range(images.shape[0]):
        minv = np.min(images_normalized[i])
        images_normalized[i] = ((images_normalized[i] - minv) /
                                (np.max(images_normalized[i]) - minv)) * 255
    
    return images_normalized


def adjust_gamma(images, gamma=1.0):
    assert (len(images.shape) == 4)
    assert (images.shape[3] == 1)

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    for i in range(images.shape[0]):
        new_images[i, :, :, 0] = cv2.LUT(np.array(images[i, :, :, 0],
                                         dtype = np.uint8), table)

    return new_images