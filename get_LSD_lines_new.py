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

        if (x1-x0)!=0:
            tan_theta = abs((y1-y0)/(x1-x0))
            if tan_theta>0.35 and tan_theta<np.tan(np.pi/3):
                continue

        #TODO 计算直线的对应的ρ、θ
        # rho = getDis(0.0,0.0,x0,y0,x1,y1);
        # if x0==x1:
        #     theta = 0
        # else:
        #     theta = np.arctan((y0-y1)/(x0-x1))
        #     if theta>0:
        #         theta = 90-theta
        #     else:
        #         thtea = 90+theta
        #
        # line_len = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        # if line_len > 450:
        #     a = 4
        #     cv2.line(new_img, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        if line_len > num_lines/1000*20:
            cv2.line(new_img, (x0, y0), (x1, y1), 255, 1, cv2.LINE_AA)
        #cv2.line(new_img, (x0, y0), (x1,y1), 255, 1, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    new_img = cv2.dilate(new_img, kernel)
    new_img = cv2.erode(new_img,kernel)
    new_img = (morphology.skeletonize(new_img//255)+0)*255.0
    #new_img = (new_img+0)*255.0
    return new_img

#点到直线的距离
def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis