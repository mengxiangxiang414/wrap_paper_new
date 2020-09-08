import os
import cv2
import numpy as np
from transform import cvtColor, warpAffine, computer_width_height
from corner_detection import detect_corners
from corner_detection1 import detect_corners1

import time

if __name__ == "__main__":
    # filename = "000026.jpg"
    filename= '4.jpg'
    # filename= '001201.jpg'
    # filename= '001402.jpg'
    # filename = '001552.jpg'
    path = os.path.join('g:/wrap/lap0', filename)
    path2 = os.path.join('g:/wrap/img2', filename)
    src_img = cv2.imread(path)
    src_img2 = cv2.imread(path)
    src_img3 = cv2.imread(path)
    #src_img2 = cv2.imread(path2)
    #gray_img = cvtColor(src_img)
    gray_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    gray_imgb = src_img[:,:,0]
    gray_imgg = src_img[:, :, 1]
    gray_imgr = src_img[:, :, 2]
    #gray_img2 = cv2.cvtColor(src_img2, cv2.COLOR_BGR2GRAY)
    # 检测角点：从左上角开始，顺时针

    time1 = time.time()
    detect_imgb, detected_cornerb = detect_corners(gray_imgb,"b0")
    detect_imgg, detected_cornerg = detect_corners(gray_imgg, "g0")
    detect_imgr, detected_cornerr = detect_corners(gray_imgr, "r0")
    #print(detected_corner)
    # for i in range(4):
    #     cv2.circle(src_img,(detected_corner[i,1],detected_corner[i,0],),3,(0,0,255),2)
    detected_corner = np.array(detected_cornerb)
    cv2.line(src_img, (detected_corner[0, 1], detected_corner[0, 0],),(detected_corner[1, 1], detected_corner[1, 0],),(0, 0, 255), 2)
    cv2.line(src_img, (detected_corner[1, 1], detected_corner[1, 0],),(detected_corner[2, 1], detected_corner[2, 0],), (0, 0, 255), 2)
    cv2.line(src_img, (detected_corner[2, 1], detected_corner[2, 0],),(detected_corner[3, 1], detected_corner[3, 0],), (0, 0, 255), 2)
    cv2.line(src_img, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],), (0, 0, 255), 2)
    cv2.imshow("b", src_img)

    detected_corner = np.array(detected_cornerg)
    cv2.line(src_img2, (detected_corner[0, 1], detected_corner[0, 0],), (detected_corner[1, 1], detected_corner[1, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img2, (detected_corner[1, 1], detected_corner[1, 0],), (detected_corner[2, 1], detected_corner[2, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img2, (detected_corner[2, 1], detected_corner[2, 0],), (detected_corner[3, 1], detected_corner[3, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img2, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],),
             (0, 0, 255), 2)
    cv2.imshow("g", src_img2)

    detected_corner = np.array(detected_cornerr)
    cv2.line(src_img3, (detected_corner[0, 1], detected_corner[0, 0],), (detected_corner[1, 1], detected_corner[1, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img3, (detected_corner[1, 1], detected_corner[1, 0],), (detected_corner[2, 1], detected_corner[2, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img3, (detected_corner[2, 1], detected_corner[2, 0],), (detected_corner[3, 1], detected_corner[3, 0],),
             (0, 0, 255), 2)
    cv2.line(src_img3, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],),
             (0, 0, 255), 2)

    top = max([detected_corner[0, 0],detected_corner[1, 0]])+10
    left = max([detected_corner[0, 1],detected_corner[3, 1]])+10
    bottom = min([detected_corner[2, 0],detected_corner[3, 0]])-10
    right = min([detected_corner[1, 1],detected_corner[2, 1]])-10
    p1,p2 = (2*left,2*top),(2*right,2*bottom)
    cv2.imshow("r", src_img3)
    #根据第一张图片的结果，检测第二张
    # detect_img2, detected_corner2 = detect_corners1(gray_img2,p1,p2)
    # detected_corner = np.array(detected_corner2)
    # cv2.line(src_img2, (detected_corner[0, 1], detected_corner[0, 0],), (detected_corner[1, 1], detected_corner[1, 0],),
    #          (0, 0, 255), 2)
    # cv2.line(src_img2, (detected_corner[1, 1], detected_corner[1, 0],), (detected_corner[2, 1], detected_corner[2, 0],),
    #          (0, 0, 255), 2)
    # cv2.line(src_img2, (detected_corner[2, 1], detected_corner[2, 0],), (detected_corner[3, 1], detected_corner[3, 0],),
    #          (0, 0, 255), 2)
    # cv2.line(src_img2, (detected_corner[3, 1], detected_corner[3, 0],), (detected_corner[0, 1], detected_corner[0, 0],),
    #          (0, 0, 255), 2)
    # #cv2.rectangle(src_img,(left,top),(right,bottom),(0,255,0),2)
    #
    # cv2.imshow("ddd2",src_img2)
    #cv2.imwrite(os.path.join('d:/wrap/imwrap', filename), src_img)
    cv2.waitKey(0)
    # time2 = time.time()
    # print("detect_corners:{}".format(time2-time1))
    # # print(detect_img.shape)
    # print(detected_corner)
    # y_max,x_max = computer_width_height(detected_corner)
    #
    # pts1 = np.array([[detected_corner[0, 1], detected_corner[0, 0]],
    #                    [detected_corner[1, 1], detected_corner[1, 0]],
    #                    [detected_corner[2, 1], detected_corner[2, 0]],
    #                    [detected_corner[3, 1], detected_corner[3, 0]]])
    # pts1 = np.float32(pts1)
    # base_points = np.array([[0, 0],
    #                         [x_max, 0],
    #                         [x_max, y_max],
    #                         [0, y_max]])
    # pts2 = np.float32(base_points)
    #
    # # 获取透视变换矩阵M，并根据M讲原图im1进行变换
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst = cv2.warpPerspective(src_img, M, (x_max, y_max))
    # cv2.imshow('image', dst)
    # # tar_img = warpAffine(src_img, detected_corner, (504, 378))
    # # # time3 = time.time()
    # # # print("warpAffine:{}".format(time3 - time2))
    # # cv2.imshow("sss",np.uint8(tar_img))

    #cv2.imwrite(os.path.join('d:/wrap/imwrap', filename), dst)
    #cv2.waitKey(0)
