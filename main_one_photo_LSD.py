import os
import cv2
import numpy as np
from transform import cvtColor, warpAffine, computer_width_height
from corner_detection_LSD import detect_corners
from corner_detection1 import detect_corners1


import time

if __name__ == "__main__":

    filename = '10.jpg'
    time1 = time.time()
    path = os.path.join('g:/wrap/img2', filename)
    src_img = cv2.imread(path)
    gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    gray_img_ori = np.copy(gray_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #gray_img = cv2.dilate(gray_img, kernel)
    # 腐蚀图像
    #gray_img = cv2.erode(gray_img, kernel)
    #gray_img = cv2.dilate(gray_img, kernel)
    grad_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    detect_img, detected_corner = detect_corners(gray_img, 12, 10, 40, 2, 2, 20, 0.7)
    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break

    detected_corner = np.array(detected_corner)
    draw_img = src_img.copy()
    cv2.line(draw_img, (detected_corner[0, 1], detected_corner[0, 0],),(detected_corner[1, 1], detected_corner[1, 0],),(0, 0, 255), 2)
    cv2.line(draw_img, (detected_corner[1, 1], detected_corner[1, 0],),(detected_corner[2, 1], detected_corner[2, 0],), (0, 0, 255), 2)
    cv2.line(draw_img, (detected_corner[2, 1], detected_corner[2, 0],),(detected_corner[3, 1], detected_corner[3, 0],), (0, 0, 255), 2)
    cv2.line(draw_img, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],), (0, 0, 255), 2)
    time2 = time.time()
    print('time===',str(time2-time1))
    cv2.imshow("draw_",draw_img)
    #cv2.imwrite('g:/wrap/books3/'+filename,draw_img)
    cv2.waitKey(0)



