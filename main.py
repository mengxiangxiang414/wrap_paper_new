import os
import cv2
import numpy as np
from transform import cvtColor, warpAffine, computer_width_height
from corner_detection import detect_corners
from corner_detection1 import detect_corners1


import time

if __name__ == "__main__":

    filename= '6.jpg'
    path = os.path.join('g:/wrap/books', filename)
    src_img = cv2.imread(path)
    gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    gray_img_ori = np.copy(gray_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 腐蚀图像
    #gray_img = cv2.erode(gray_img, kernel)
    gray_img = cv2.dilate(gray_img, kernel)

    # hsv = cv2.cvtColor(src_img,cv2.COLOR_BGR2HSV)
    # gray_img = hsv[:,:,0]
    #滑动条
    def callback(object):
        pass
    cv2.namedWindow('image')
    cv2.createTrackbar('top_lines', 'image', 26,60, callback)
    cv2.createTrackbar('1:laplacian,2:sobel,3:canny', 'image', 3, 3, callback)
    cv2.createTrackbar('low_threshold', 'image', 10, 127, callback)
    cv2.createTrackbar('up_threshold', 'image', 40, 255,callback)
    cv2.createTrackbar('theta', 'image', 2, 8, callback)
    cv2.createTrackbar('rho', 'image', 1, 8, callback)
    cv2.createTrackbar('corner_area', 'image', 20, 50, callback)
    cv2.createTrackbar('vote_rate', 'image', 8, 10, callback)

    # 1. Shi-Tomasi角点检测
    # 优点：速度相比Harris有所提升，可以直接得到角点坐标
    # corners = cv2.goodFeaturesToTrack(gray_img_ori, 1000, 0.05, 5)
    # corners = np.int0(corners)
    #
    # for i in corners:
    #     # 压缩至一维：[[62, 64]] -> [62, 64]
    #     x, y = i.ravel()
    #     cv2.circle(src_img, (x, y), 4, (255, 0, 0), 2)
    #cv2.imshow("111", src_img)

    while (True):

        top_lines = cv2.getTrackbarPos('top_lines', 'image')
        grad_option = cv2.getTrackbarPos('1:laplacian,2:sobel,3:canny', 'image')
        low_threshold = cv2.getTrackbarPos('low_threshold', 'image')
        up_threshold = cv2.getTrackbarPos('up_threshold', 'image')
        theta = cv2.getTrackbarPos('theta', 'image')
        rho = cv2.getTrackbarPos('rho', 'image')
        corner_area = cv2.getTrackbarPos('corner_area', 'image')
        vote_rate = cv2.getTrackbarPos('vote_rate', 'image')

        if top_lines<2 or low_threshold>=up_threshold:
            continue
        detect_img, detected_corner = detect_corners(gray_img, top_lines, grad_option, low_threshold, up_threshold, theta, rho, corner_area, gray_img_ori, vote_rate/10)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

        detected_corner = np.array(detected_corner)
        draw_img = src_img.copy()
        cv2.line(draw_img, (detected_corner[0, 1], detected_corner[0, 0],),(detected_corner[1, 1], detected_corner[1, 0],),(0, 0, 255), 2)
        cv2.line(draw_img, (detected_corner[1, 1], detected_corner[1, 0],),(detected_corner[2, 1], detected_corner[2, 0],), (0, 0, 255), 2)
        cv2.line(draw_img, (detected_corner[2, 1], detected_corner[2, 0],),(detected_corner[3, 1], detected_corner[3, 0],), (0, 0, 255), 2)
        cv2.line(draw_img, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],), (0, 0, 255), 2)
        cv2.imshow("draw_",draw_img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

