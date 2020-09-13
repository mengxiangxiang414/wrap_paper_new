import cv2
import numpy as np
from skimage import morphology
import math
import time

def get_LSD(gray_img, vote_table):
    #cv2.imshow('gray_imggg',gray_img)
    h, w = gray_img.shape
    new_img = np.zeros([h, w])
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果
    dlines = lsd.detect(gray_img)
    # 绘制检测结果
    num_lines = dlines[0].size


    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))

        line_len = int(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

        # TODO 计算直线的对应的ρ、θ
        rho = getDis(0.0, 0.0, x0, y0, x1, y1)
        if x0 == x1:
            theta = 0
        else:
            theta = np.arctan((y0 - y1) / (x0 - x1))
            theta = theta * 180 / np.pi
            theta = 90 + theta

        theta_floor = int(np.floor(theta))
        theta_ceil = int(np.ceil(theta))
        rho_floor = int(np.floor(rho / 2) + (n + 1) // 2)
        rho_ceil = int(np.ceil(rho / 2) + (n + 1) // 2)

        np.add.at(vote_table, (theta_floor, rho_floor), line_len)
        np.add.at(vote_table, (theta_ceil, rho_floor), line_len)
        np.add.at(vote_table, (theta_floor, rho_ceil), line_len)
        np.add.at(vote_table, (theta_ceil, rho_ceil), line_len)
    # return new_img

#点到直线的距离
def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis