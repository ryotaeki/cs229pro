import dlib
from imutils import face_utils
import cv2

face_detector = dlib.get_frontal_face_detector()

predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)
for i in range(15000):
    if i % 500 == 0:
        print("processing ", i)
    input_img_name = "dataset/trainB/" + str(i) + ".png"
    img = cv2.imread(input_img_name)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(img_gry, 1)

    if len(faces) > 0:
        landmark = face_predictor(img_gry, face[0])
        landmark = face_utils.shape_to_np(landmark)
        left_x, left_y = landmark[39]
        right_x, right_y = landmark[42]
        angle = face_angle(left_x, left_y, right_x, right_y)
        angles = np.full((256, 256, 1), angle)
        new_data = np.concatenate([img, angles], axis=2)
        np.save("dataset/newtrainB/" + str(i), new_data)
