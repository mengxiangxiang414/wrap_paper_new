import os
import cv2
import numpy as np
from transform import cvtColor, warpAffine, computer_width_height
from corner_detection_LSD import detect_corners
import time

def process(path):
    ratio = 210 / 297
    time1 = time.time()

    src_img = cv2.imread(path)
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    detect_img, detected_corner = detect_corners(gray_img, 12, 10, 40, 2, 1, 20,
                                                 0.6)  # left_top, right_top, right_bottom, left_bottom

    pts2 = np.array(detected_corner)


    len_1 = (pts2[0, 0] - pts2[1, 0]) ** 2 + (pts2[0, 1] - pts2[1, 1]) ** 2
    len_2 = (pts2[1, 0] - pts2[2, 0]) ** 2 + (pts2[1, 1] - pts2[2, 1]) ** 2
    len_3 = (pts2[2, 0] - pts2[3, 0]) ** 2 + (pts2[2, 1] - pts2[3, 1]) ** 2
    len_4 = (pts2[3, 0] - pts2[1, 0]) ** 2 + (pts2[3, 1] - pts2[1, 1]) ** 2

    len_max = np.sqrt(max([len_1, len_2, len_3, len_4]))
    y_max = np.uint16(len_max)
    x_max = np.uint16(y_max * ratio)

    # 透视变换的四个参考点
    if (len_1 + len_3 < len_2 + len_4):
        base_points = np.array([[0, 0],
                                [x_max - 1, 0],
                                [x_max - 1, y_max - 1],
                                [0, y_max - 1]])
    else:
        base_points = np.array([[0, 0],
                                [y_max - 1, 0],
                                [y_max - 1, x_max - 1],
                                [0, x_max - 1]])
    # 获取透视变换矩阵M，并根据M讲原图im1进行变换
    pts2 = np.float32(pts2)
    pts2 = pts2[:, [1, 0]]
    base_points = np.float32(base_points)
    M = cv2.getPerspectiveTransform(pts2, base_points)
    dst = cv2.warpPerspective(src_img, M, (x_max, y_max))

    # 锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    dst2 = cv2.filter2D(dst, -1, kernel=kernel)
    # cv2.imwrite(path, dst2)
    return list(dst2)

if __name__ == "__main__":
    dirs = 'data2'
    for imgname in os.listdir(dirs):
        imgpath = dirs+'/'+imgname
        # process(imgpath)
        time1 = time.time()
        process(imgpath)
        time2 = time.time()
        print(imgname+': {}'.format(time2-time1))

