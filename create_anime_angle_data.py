import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from CFA import CFA
from face_angle import face_angle

# param
img_width = 256
checkpoint_name = 'checkpoint_landmark_191116.pth.tar'

# detector
face_detector = cv2.CascadeClassifier('lbpcascade_animeface.xml')
landmark_detector = CFA(output_channel_num=num_landmark + 1, checkpoint_name=checkpoint_name).cuda()

for i in range(15000):
    if i % 500 == 0:
        print("processing ", i)
    input_img_name = "dataset/trainA/" + str(i) + ".png"
    img = cv2.imread(input_img_name)
    faces = face_detector.detectMultiScale(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(face) > 0:
        (x_, y_, w_, h_) = faces[0]

        x = max(x_ - w_ / 8, 0)
        rx = min(x_ + w_ * 9 / 8, img.width)
        y = max(y_ - h_ / 4, 0)
        by = y_ + h_
        w = rx - x
        h = by - y

        img_tmp = img.crop((x, y, x+w, y+h))
        img_tmp = img_tmp.resize((img_width, img_width), Image.BICUBIC)
        img_tmp = train_transform(img_tmp)
        img_tmp = img_tmp.unsqueeze(0).cuda()

        # estimate heatmap
        heatmaps = landmark_detector(img_tmp)
        heatmaps = heatmaps[-1].cpu().detach().numpy()[0]

        heatmaps_tmp = cv2.resize(heatmaps[14], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width
        left_x = x + landmark_x
        left_y = y + landmark_y

        heatmaps_tmp = cv2.resize(heatmaps[19], (img_width, img_width), interpolation=cv2.INTER_CUBIC)
        landmark = np.unravel_index(np.argmax(heatmaps_tmp), heatmaps_tmp.shape)
        landmark_y = landmark[0] * h / img_width
        landmark_x = landmark[1] * w / img_width
        right_x = x + landmark_x
        right_y = y + landmark_y

        angle = face_angle(left_x, left_y, right_x, right_y)
        angles = np.full((256, 256, 1), angle)
        new_data = np.concatenate([img, angles], axis=2)
        np.save("dataset/newtrainA/" + str(i), new_data)
