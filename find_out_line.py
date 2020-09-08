import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue


min_num = 50
#[2,3]
def getPoints(p1,p2,width):
    points = []
    pxs = []
    pys = []
    x1,y1 = p1
    x2,y2 = p2
    for i in range(width):
        y3 = (y2 - y1) / (x2 - x1) * (i - x1) + y1
        points.append([int(y3),int(i)])
        pxs.append(int(i))
        pys.append(int(y3))
    return points,pxs,pys

#对一个像素，判断其是否是左下角
def check_bottom_left(img,p,flag,h,w,origin_angle,min_num):
    maxsize = 2000
    q = Queue(maxsize)
    q.put(p)
    count = 0
    while not q.empty():
        m,n = q.get()
        if n==34:
            dd = 0
        for i in range(-1, 0, 1):
            for j in range(-1, 2, 1):
                if m+i>0 and m+i<h and n+j>0 and n+j<w:
                    # if flag[m+i,n+j]==1:
                    #     continue
                    # flag[m+i,n+j] = 1
                    if i==0 and j==0:
                        continue

                    if img[m+i,n+j]<255:
                        continue
                    if (m+i-p[0])==1 and abs(n+j-p[1])==1 or (m+i-p[0])==2 and abs(n+j-p[1])==1:
                        q.put([m+i,n+j])
                        count += 1
                    else:
                        if n+j-p[1]==0:
                            q.put([m + i, n + j])
                            count += 1
                        else:
                            this_tan = (m+i-p[0])/(n+j-p[1])#当前像素点和原始点的连线  与  水平线的夹角
                            this_angle = np.arctan(this_tan)
                            delta_angle = abs(this_angle-origin_angle)
                            if delta_angle > 1.5:
                                delta_angle = np.pi-delta_angle
                            if delta_angle> 1.1:
                                q.put([m+i,n+j])
                                count += 1
    if count<min_num:
        print('个数：',str(count))
        return False

    return True

def find_left_points(img,p1,p2,p3,p4,left_left_lines):
    #left-top
    points = getPoints(p1,p2,p1[1])
    for pt in points:
        top_edge = pt[0] - 2 if pt[0] - 2 >= 0 else 0
        left_edge = pt[1] - 2 if pt[1] - 2 >= 0 else 0
        plt.imshow(img[top_edge:pt[0] + 3, left_edge:pt[1] + 3])
        plt.grid()
        plt.show()
        is_right_corner = check_bottom_left(img, pt, flags, height, width, origin_angle, min_num)
        print('x={}is {}'.format(pt[1], is_right_corner))
        if is_right_corner:
            print(pt)
    #left-bottom

    return
def find_right_points():
    return
def find_top_points():
    return
def find_bottom_points():
    return
def find_final_points(grad_img,p1,p2,p3,p4,left_left_lines,top_top_lines,right_right_lines,bottom_bottom_lines):
    if len(left_left_lines)>0:
        left_points = find_left_points(grad_img,p1,p2,p3,p4,left_left_lines)
    if len(right_right_lines)>0:
        left_points = find_right_points()
    if len(top_top_lines)>0:
        left_points = find_top_points()
    if len(bottom_bottom_lines)>0:
        left_points = find_bottom_points()
    return



# src_path = 'h:/grad.png'
# img = cv2.imread(src_path,0)
# cv2.imshow('before',img)
#
#
#
#
# height,width = img.shape[0],img.shape[1]
# p1 = [137,474]
# p2 = [625,452]
#
# origin_angle = np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))
#
# points,pxs,pys = getPoints(p1,p2,p1[0])
# # img[pys,pxs] = 255
# # cv2.imshow('after',img)
#
# flags = np.zeros(img.shape)
# for pt in points:
#     top_edge = pt[0]-2 if pt[0]-2>=0 else 0
#     left_edge = pt[1] - 2 if pt[1] - 2 >= 0 else 0
#     plt.imshow(img[top_edge:pt[0] + 3, left_edge:pt[1] + 3])
#     plt.grid()
#     plt.show()
#     is_right_corner = check_bottom_left(img,pt,flags,height,width,origin_angle,min_num)
#     print('x={}is {}'.format(pt[1],is_right_corner))
#     if is_right_corner:
#         print(pt)
# cv2.waitKey(0)

#for i in range(width):




