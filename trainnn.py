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


#2 各種設定

train_data_path = 'dataset' # ここを変更。Colaboratoryにアップロードしたzipファイルを解凍後の、データセットのフォルダ名を入力

image_size = 64 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。

color_setting = 3  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。

#3 データセットの読み込みとデータ形式の設定・正規化・分割

A_image = []
B_image = []

A_path = glob.glob("dataset/nnA/*")
B_path = glob.glob("dataset/nnB/*")

for file in A_path:
  img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
  array = img_to_array(img)
  A_image.append(array)


for file in B_path:
  img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
  array = img_to_array(img)
  B_image.append(array)

A_image = np.array(A_image)
B_image = np.array(B_image)

#X_image = X_image.astype('float32') / 255

trainA_images, validA_images, trainB_images, validB_images = train_test_split(A_image, B_image, test_size=0.10)
A_train = trainA_images
B_train = trainB_images
A_test = validA_images
B_test = validB_images


#4 機械学習（人工知能）モデルの作成 – 畳み込みニューラルネットワーク（CNN）・学習の実行等

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
plot_model(model, to_file='model.png')

model.compile(loss='mean_squared_error',
              optimizer=Adam(),
              metrics=['mse'])

start_time = time.time()

# ここを変更。必要に応じて「batch_size=」「epochs=」の数字を変更してみてください。
history = model.fit(A_train,B_train, batch_size=100, epochs=15, verbose=1, validation_data=(A_test, B_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

open('cnn_model.json','w').write(model.to_json())
model.save_weights('cnn_weights.h5')
#model.save('cnn_model_weight.h5') #モデル構造と重みを1つにまとめることもできます

score = model.evaluate(A_test, B_test, verbose=0)
print('Loss:', score[0], '（損失関数値 - 0に近いほど正解に近い）')
print('Accuracy:', score[1] * 100, '%', '（精度 - 100% に近いほど正解に近い）')
print('Computation time（計算時間）:{0:.3f} sec（秒）'.format(time.time() - start_time))
