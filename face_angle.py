import numpy as np

def face_angle(left_x, left_y, right_x, right_y):
    if right_x - left_x == 0:
        if right_y > left_y:
            return 90 * (256/180)
        else:
            return -90 * (256/180)
    else:
        deg = np.rad2deg(np.arctan((right_y - left_y)/(right_x - left_x)))
        return deg * (256/180)
