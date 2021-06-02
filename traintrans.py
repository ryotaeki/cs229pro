import cv2
import glob
import os
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

data_size = 15000
batch_size = 100
max_iter = 10000
lr = 0.1

trainA_path = glob.glob("dataset/trainA/*")
trainB_path = glob.glob("dataset/trainB/*")
img_A = []
img_B = []

for i in range(data_size):
    if i % 200 == 0:
        print("preprocessing ", i)
    img_A.append(img_to_array(Image.open("dataset/trainA/" + str(i) + ".png")))
    img_B.append(img_to_array(Image.open("dataset/trainB/" + str(i) + ".png")))

trans_vector = np.zeros((256, 256, 3))
for t in range(1, max_iter):
    if t % 200 == 0:
        print("processing ", t)
    trainA_id = np.random.randint(0, data_size, batch_size)
    trainB_id = np.random.randint(0, data_size, batch_size)
    ave_A = np.zeros((256, 256, 3))
    ave_B = np.zeros((256, 256, 3))
    for id in trainA_id:
        ave_A += img_A[id]
    ave_A /= batch_size
    for id in trainB_id:
        ave_B += img_B[id]
    ave_B /= batch_size
    diff = (ave_B - ave_A + trans_vector)
    trans_vector += lr * diff
    if t % 1000 == 0:
        np.save("trans_vector_" + str(t), trans_vector)
np.save("trans_vector_10000", trans_vector)
