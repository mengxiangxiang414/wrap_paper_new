import cv2
import os
from queue import Queue
import numpy as np

def track_p(q, grad_img,flags,m, n, min_threshold):
    h, w = grad_img.shape
    count = 1 #统计联通像素个数
    flags[m, n] = 1
    q.put([m, n])
    link_points_h = []
    link_points_w = []
    link_points_h.append(m)
    link_points_w.append(n)
    while not q.empty():
        [m, n] = q.get()

        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if m+i>=0 and m+i<h and n+j>0 and n+j<w:
                    if i == 0 and j == 0:
                        continue
                    if flags[m+i][n+j] == 1:
                        continue
                    if grad_img[m+i][n+j] == 0:
                        flags[m+i][n+j] = 1
                        continue
                    flags[m + i, n + j] = 1
                    q.put([m + i, n + j])
                    link_points_h.append(m + i)
                    link_points_w.append(n + j)
                    count += 1
    if count<min_threshold:
        grad_img[link_points_h,link_points_w] = 0

def clear_grad_function(grad_img, min_threshold):
    h, w = grad_img.shape
    flags = np.zeros([h, w])
    maxsize = 2000
    q = Queue(maxsize)
    for i in range(h):
        for j in range(w):
            if flags[i][j] == 0:#未被访问过
                if grad_img[i][j] > 0:#是边缘点
                    #进行判断该像素
                    track_p(q,grad_img,flags,i,j, min_threshold)
                else:
                    flags[i][j] = 1
    return grad_img



