import cv2
import glob
import os
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

data_size = 100

for i in range(data_size):
    print(i - 1, " finished")
    img = img_to_array(Image.open("dataset/testA/" + str(i) + ".png"))
    for iter in range(1, 11):
        num = iter * 1000
        trans_vector = np.load("trans_vector_" + str(num) + ".npy")
        pred = img + trans_vector
        pred = np.clip(pred, 0, 255)
        image = Image.fromarray(pred.astype(np.uint8))
        image.save("dataset/resulttrans/" + str(num) + "/" + str(i) + ".png")
