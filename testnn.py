import keras
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten
#from keras.utils import plot_model
#from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
from PIL import Image


#2 各種設定

train_data_path = 'dataset' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = 64 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
          input_shape=(image_size, image_size, color_setting), activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2DTranspose(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2DTranspose(16, (3, 3), padding='same', activation='relu'))
model.add(Conv2DTranspose(3, (3, 3), padding='same', activation='relu'))
#model.add(Dense(class_number, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mse'])
model.load_weights("cnn_weights.h5")

for i in range(100):
    img = Image.open('dataset/nntestA/' + str(i) + '.png')
    array = img_to_array(img).reshape((-1, 64, 64, 3))
    pred = model.predict(array)[0]
    pred = np.clip(pred, 0, 255)
    image = Image.fromarray(pred.astype(np.uint8))
    image.save("dataset/resultnn/" + str(i) + ".png")
