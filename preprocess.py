from PIL import Image
import os
import glob

for i in range(100):
    img = Image.open('dataset/testA/' + str(i) + '.png')
    img_resize = img.resize((256, 256))
    img_resize.save('dataset/testA/' + str(i) + '.png')
