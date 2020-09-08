import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from utils import Conv2d
from lines_cross import *

'''
可调参数
@get_grad_img: grad阈值
'''


def filter2D(img, kernel):
    #     计算需要padding的大小
    assert kernel.shape[0] == kernel.shape[1]
    pad_size = int(kernel.shape[0] / 2)  # 每个边应该padding的厚度
    kernel_size = kernel.shape[0]
    row, col = img.shape

    # TODO: 用边缘像素点填充
    # top_padding_content = np.tile(img[0, :], (pad_size, 1))
    # bottom_padding_content = np.tile(img[-1, :], (pad_size, 1))
    # print(top_padding_content.shape)
    # left_padding_content = np.tile(img[:, 0], (1, pad_size  ))
    # right_padding_content = np.tile(img[:, -1], (1, pad_size ))
    # img = np.vstack((top_padding_content, img, bottom_padding_content))
    # img = np.hstack((left_padding_content, img, right_padding_content))

    # 用0填充
    top_padding_content = np.zeros((pad_size, col))
    left_padding_content = np.zeros((row + 2 * pad_size, pad_size))
    img = np.vstack((top_padding_content, img, top_padding_content))
    img = np.hstack((left_padding_content, img, left_padding_content))
    grad_img = np.zeros((row, col))
    for i in range(img.shape[0] - kernel_size + 1):  # new_size- filter_size+1 @important
        for j in range(img.shape[1] - kernel_size + 1):
            img_part = img[i:i + kernel_size, j:j + kernel_size]
            result = np.sum(
                img_part * kernel
            )
            grad_img[i][j] = result
    # 边界平滑
    # done: 能够适应任何size的kernel
    top_padding_content = np.tile(grad_img[pad_size, :], (pad_size, 1))
    grad_img[0:pad_size, :] = top_padding_content

    bottom_padding_content = np.tile(grad_img[-(pad_size + 1), :], (pad_size, 1))
    grad_img[-pad_size:, :] = bottom_padding_content

    left_padding_content = np.tile(np.expand_dims(grad_img[:, pad_size], axis=1), (1, pad_size))
    grad_img[:, 0:pad_size] = left_padding_content

    top_padding_content = np.tile(np.expand_dims(grad_img[:, -(pad_size + 1)], axis=1), (1, pad_size))
    grad_img[:, -pad_size:] = top_padding_content

    # top_padding_content = grad_img[1, :]
    # grad_img[0, :] = top_padding_content
    # top_padding_content = grad_img[-2, :]
    # grad_img[-1, :] = top_padding_content
    # top_padding_content = grad_img[:, 1]
    # grad_img[:, 0] = top_padding_content
    # top_padding_content = grad_img[:, -2]
    # grad_img[:, -1] = top_padding_content
    return grad_img


def make_gauss_filter(size, std_D):
    assert size % 2 == 1
    start = int(size / 2)
    x = np.arange(-start, start + 1)
    x = np.ravel(np.tile(x, (size, 1)))
    y = np.arange(start, -start - 1, step=-1)
    y = np.repeat(y, size)
    power = -(x * x + y * y) / (2 * std_D * std_D)
    filter = np.exp(power) / (2 * np.pi * std_D * std_D)
    filter = filter / np.sum(filter)
    return np.reshape(filter, (size, size))


def get_grad_img2(gray_img):
    filter_size = 5
    gauss = make_gauss_filter(filter_size, 1.1)
    # smoothed_img = filter2D(gray_img, kernel=gauss)

    filter2D=Conv2d(filter_size,1,1,1,weight=gauss[np.newaxis,:],mode='valid')
    smoothed_img = filter2D(gray_img[np.newaxis,:])[0]
    row_ind, col_ind = np.where(smoothed_img > 255)
    smoothed_img[row_ind, col_ind] = 255
    # 显示灰度图
    # cv2.imshow('wind', np.uint8(smoothed_img))
    # cv2.waitKey(0)
    time11 = time.time()
    laplace = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )
    # grad_img = filter2D(smoothed_img, kernel=laplace)
    filter2D = Conv2d(3, 1, 1, 1, weight=laplace[np.newaxis, :],mode='valid')
    grad_img=filter2D(smoothed_img[np.newaxis,:])[0]
    # plt.imshow(grad_img)
    # plt.show()

    grad_img = np.where(grad_img > 15, grad_img, 0)
    # plt.imshow(grad_img,cmap="gray")
    # plt.show()
    cv2.imshow("hhhh", grad_img)
    return grad_img
def get_grad_img(gray_img):
    img_shape = gray_img.shape
    smoothed_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    #grad_img = cv2.Laplacian(smoothed_img, cv2.CV_8U, ksize=3)
    laplace = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )
    laplace2 = np.array(
        [
            [-2, 0, -2],
            [0, 8, 0],
            [-2, 0, -2]
        ],
        dtype=np.float32
    )
    laplace3 = np.array(
        [
            [-2, 0, -2],
            [0, 8, 0],
            [-2, 0, -2]
        ],
        dtype=np.float32
    )
    grad_img = cv2.filter2D(smoothed_img, -1, laplace)
    #grad_img = cv2.filter2D(grad_img, -1, laplace2)
    grad_img = np.where(grad_img > 5, grad_img, 0)
    # plt.imshow(grad_img,cmap="gray")
    # plt.show()
    cv2.imshow("hh",grad_img)
    return grad_img
def get_grad_img3(gray_img):
    img_shape = gray_img.shape
    time1 = time.time()
    im = cv2.GaussianBlur(gray_img, (5, 5), 0)
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    # 分别为计算x和y，再求和
    sobelxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 将x和y融合起来
    grad_img = np.where(sobelxy > 30, sobelxy, 0)
    time2 = time.time()
    # plt.imshow(grad_img)
    # plt.show()
    return grad_img[3:img_shape[0]-3,3:img_shape[1]-3]

def houghLines(grad_img):
    """
    从梯度图进行hough变换，检测直线
    :param grad_img: 梯度图
    :return: 检测出来的直线的极坐标表示
    """
    # --------------------------------------投票------------------------------------------
    rho_max = math.sqrt(grad_img.shape[0] * grad_img.shape[0] + grad_img.shape[1] * grad_img.shape[1])
    time1 = time.time()
    m, n = 180, int(rho_max)
    theta_range = np.linspace(0, np.pi, m)
    rho_range = np.linspace(-rho_max, rho_max, n)
    # 投票的表格
    vote_table = np.zeros(shape=(m, n))

    row_cor, col_cor = np.where(grad_img > 0)  # 挑出有选举权的点,假设有K个
    cor_mat = np.stack((row_cor, col_cor), axis=1)  # K*2
    K = cor_mat.shape[0]

    cos_theta = np.cos(theta_range)
    sin_theta = np.sin(theta_range)
    # 这是一个大坑，row实际对应的是y
    # theta_mat = np.stack((cos_theta, sin_theta), axis=0)  # 2*m
    theta_mat = np.stack((sin_theta, cos_theta), axis=0)  # 2*m

    y_mat = np.matmul(cor_mat, theta_mat)  # K*m

    rho_ind = (
            (y_mat - (-rho_max)) * (n - 1) / (rho_max - (-rho_max))
    ).astype(np.int32)  # K*m
    rho_ind = np.ravel(rho_ind, order='F')  # 在列方向stack

    theta_ind = np.arange(0, m)[:, np.newaxis]
    theta_ind = np.repeat(theta_ind, K)


    np.add.at(vote_table, (theta_ind, rho_ind), 1)  # 在vote_table中投票
    time2 = time.time()
    print("pre:{}".format(time2 - time1))
    # ----------------------------------过滤 1： 选出不同的直线-------------------------------
    # 取出top_k条不同的直线
    top_k = 10
    top_vert = 0
    top_hori = 0
    # unravel_index: https://www.jianshu.com/p/a7e19847bd39 -> 将一维index转换到(m,n)维上
    argmax_ind = np.dstack(np.unravel_index(np.argsort(-vote_table.ravel(), ), (m, n)))
    time3 = time.time()
    print("top:{}".format(time3 - time2))
    argmax_ind = argmax_ind[0, :, :]
    valid_lines = np.zeros((top_k, 2))
    exist_num = 0


    for i in range(0, m * n):
        row_ind, col_ind = tuple(argmax_ind[i])
        theta = theta_range[row_ind]
        rho = rho_range[col_ind]
        if theta==0 and rho==0:
            continue
        if is_new_line(theta, rho, valid_lines, exist_num):
            # 遇到新的线了
            #水平的、竖直的每一种都至少取5个
            is_vert_line = np.abs(theta - 1.5) > 0.5
            if is_vert_line:
                if top_vert<(top_k//2):
                    valid_lines[exist_num][0] = theta
                    valid_lines[exist_num][1] = rho
                    top_vert += 1
                    exist_num += 1
            else :
                if top_hori<(top_k//2):
                    valid_lines[exist_num][0] = theta
                    valid_lines[exist_num][1] = rho
                    top_hori += 1
                    exist_num += 1
            if top_hori>=(top_k//2) and top_vert>=(top_k//2):
                break

            # if exist_num == 4:
            #     area, points = get_area(valid_lines)
            #     out_nums = 0
            #     if points[0][0]< -50:
            #         print(points)
            #     for i in range(4):
            #         out_bd = out_boundary(points[i],grad_img.shape)
            #         if out_bd:
            #             out_nums += 1
            #     if out_nums >= 2:
            #         exist_num -= 1
            #         continue
            #     too_small = is_too_small(area, grad_img.shape)
            #     if too_small:
            #         exist_num -= 1
            # if exist_num >= top_k:
            #     break
    return valid_lines

def out_boundary(p,shape):
    y,x = p
    h,w = shape
    if (0<x and x<w) and (0<y and y<h):
        return False
    else:
        return True

def out_boundary_too_far(p,shape):
    y, x = p
    h, w = shape
    if (-0.01*w < x and x < 1.01*w) and (-0.01*h < y and y < 1.01*h):
        return False
    else:
        return True
def is_too_small(area, shape):
    img_area = shape[0] * shape[1]
    rate = area / img_area
    # print(rate)
    if rate < 1 / 3:
        return True


def get_area(polar_lines):
    # 1. 为了化简计算,把直线分成接近水平/垂直, 两种直线
    vert_ind = np.abs(polar_lines[:, 0] - 1.5) > 0.5
    vert_lines = polar_lines[vert_ind, :]  # 接近垂直的直线
    hori_lines = polar_lines[np.logical_not(vert_ind), :]  # 接近水平的直线

    # 排序: 为了能够组成正方形,先进行排序
    test = np.argsort(np.abs(vert_lines[:, 1]))
    vert_lines = vert_lines[test, :]

    test = np.argsort(np.abs(hori_lines[:, 1]))
    hori_lines = hori_lines[test, :]

    # 2. 计算交点
    points = []
    num_vert_lines = vert_lines.shape[0]
    num_hori_lines = hori_lines.shape[0]
    for i in range(num_vert_lines):
        for j in range(num_hori_lines):
            point = get_intersection_points(vert_lines[i], hori_lines[j])
            points.append([point[1], point[0]])

    # 3. 近似面积最大的为角点
    points = np.array(points).reshape(num_vert_lines, num_hori_lines, 2)
    max_area = 0
    for i in range(num_vert_lines - 1):
        for j in range(num_hori_lines - 1):
            left_top = points[i][j]
            left_bottom = points[i][j + 1]
            right_top = points[i + 1][j]
            right_bottom = points[i + 1][j + 1]
            #area = get_approx_area(left_top, left_bottom, right_top, right_bottom)
            area = calc_area2(left_top,left_bottom,right_bottom)+calc_area2(left_top,right_top,right_bottom)
            if area > max_area:
                max_area = area
                point_seq = (left_top, right_top, right_bottom, left_bottom)
    return max_area, point_seq


def detect_corners1(gray_img,p1,p2):
    time1 = time.time()
    grad_img = get_grad_img(gray_img)

    #对梯度图进行掩膜
    grad_img[p1[1]:p2[1],p1[0]:p2[0]] = 0
    grad_img = cv2.medianBlur(grad_img, 3)
    cv2.imshow("dsdsd",grad_img)
    #cv2.imwrite("d:/wrap/img_gray/1.jpg",grad_img)

    time2 = time.time()
    print("get_grad_img:{}".format(time2 - time1))
    polar_lines = houghLines(grad_img)
    time3 = time.time()
    print("houghLines:{}".format(time3 - time2))

    # -------------------------------计算交点----------------------------------
    # 1. 为了化简计算,把直线分成接近水平/垂直, 两种直线
    vert_ind = np.abs(polar_lines[:, 0] - 1.5) > 0.5
    vert_lines = polar_lines[vert_ind, :]  # 接近垂直的直线
    hori_lines = polar_lines[np.logical_not(vert_ind), :]  # 接近水平的直线

    # 排序: 为了能够组成正方形,先进行排序
    test = np.argsort(np.abs(vert_lines[:, 1]))
    vert_lines = vert_lines[test, :]

    test = np.argsort(np.abs(hori_lines[:, 1]))
    hori_lines = hori_lines[test, :]

    #将候选直线进行排列组合，两个一组
    num_vert_lines = vert_lines.shape[0]
    num_hori_lines = hori_lines.shape[0]
    num_vert_couples = num_vert_lines*(num_vert_lines-1)//2
    num_hori_couples = num_hori_lines * (num_hori_lines - 1) //2
    couples_vert = []
    couples_hori = []
    for i in range(num_vert_lines):
        for j in range(i+1,num_vert_lines):
            couples_vert.append([i,j])

    for i in range(num_hori_lines):
        for j in range(i+1,num_hori_lines):
            couples_hori.append([i,j])

    area_table = np.zeros((num_vert_couples,num_hori_couples))
    for i in range(num_vert_couples):
        for j in range(num_hori_couples):
            #下面计算每四条线的围成的四边形的面积
            temp_vert_lines = vert_lines[couples_vert[i]]
            temp_hori_lines = hori_lines[couples_hori[j]]
            area_table[i][j] = get_area_by_four_lines(temp_vert_lines,temp_hori_lines,gray_img.shape[:2])

    #求出最大的面积，以及其所对应的直线的index
    m,n = np.unravel_index(np.argmax(area_table),(num_vert_couples,num_hori_couples))
    final_vert_lines = vert_lines[couples_vert[m]]
    final_hori_lines = hori_lines[couples_hori[n]]
    points = get_corner_by_four_lines(final_vert_lines, final_hori_lines)

    vec12 = [points[1][1] - points[0][1], points[1][0] - points[0][0]]
    vec14 = [points[3][1] - points[0][1], points[3][0] - points[0][0]]
    vec32 = [points[1][1] - points[2][1], points[1][0] - points[2][0]]
    vec34 = [points[3][1] - points[2][1], points[3][0] - points[2][0]]
    dot1 = np.dot(vec12, vec14) / (np.sqrt(vec12[0] ** 2 + vec12[1] ** 2)) / (np.sqrt(vec14[0] ** 2 + vec14[1] ** 2))
    dot2 = np.dot(vec32, vec34) / (np.sqrt(vec32[0] ** 2 + vec32[1] ** 2)) / (np.sqrt(vec34[0] ** 2 + vec34[1] ** 2))
    print(dot1)
    print(dot2)

    points = np.array(points).reshape(2, 2, 2)

    left_top = points[0][0]
    left_bottom = points[0][ 1]
    right_top = points[1][0]
    right_bottom = points[1][1]
    point_seq = (left_top, right_top, right_bottom, left_bottom)


    return grad_img, point_seq

#根据四条直线，计算所围成的面积
def get_area_by_four_lines(vert_lines,hori_lines, img_shape):
    num_vert_lines = 2
    num_hori_lines = 2

    points = get_corner_by_four_lines(vert_lines,hori_lines)

    #判断四个角点是否出界
    out_nums = 0
    temp_i = 0
    for i in range(4):
        out_bd = out_boundary(points[i],img_shape)
        if out_bd:
            temp_i = i
            out_nums += 1
    if out_nums >= 2:
        return 0    #若有两个角点在界外，则直接返回0
    #如果只有一个角出界，则判断是否离边界太远
    if out_nums==1:
        out_bd_too_far = out_boundary_too_far(points[temp_i],img_shape)
        if out_bd_too_far:
            return 0#若有两个角点在界外，则直接返回0
    #判断对着的两个角必须一个锐角一个钝角
    # vec12 = [points[1][1]-points[0][1],points[1][0]-points[0][0]]
    # vec14 = [points[3][1]-points[0][1],points[3][0]-points[0][0]]
    # vec32 = [points[1][1]-points[2][1],points[1][0]-points[2][0]]
    # vec34 = [points[3][1]-points[2][1],points[3][0]-points[2][0]]
    # dot1 = np.dot(vec12,vec14)/(np.sqrt(vec12[0]**2+vec12[1]**2))/(np.sqrt(vec14[0]**2+vec14[1]**2))
    # dot2 = np.dot(vec32,vec34)/(np.sqrt(vec32[0]**2+vec32[1]**2))/(np.sqrt(vec34[0]**2+vec34[1]**2))
    # print(dot1)
    # print(dot2)

    #判断四个交点是否交叉
    p0 = MyPoint(points[0][1],points[0][0])
    p1 = MyPoint(points[1][1],points[1][0])
    p2 = MyPoint(points[2][1], points[2][0])
    p3 = MyPoint(points[3][1], points[3][0])

    d = IsIntersec(p0, p1, p2, p3)
    if d==1:
        return 0

    # 3. 近似面积最大的为角点
    points = np.array(points).reshape(2, 2, 2)
    max_area = 0
    for i in range(num_vert_lines - 1):
        for j in range(num_hori_lines - 1):
            left_top = points[i][j]
            left_bottom = points[i][j + 1]
            right_top = points[i + 1][j]
            right_bottom = points[i + 1][j + 1]
            # area = get_approx_area(left_top, left_bottom, right_top, right_bottom)
            area = calc_area2(left_top, left_bottom, right_bottom) + calc_area2(left_top, right_top, right_bottom)
            if area > max_area:
                max_area = area
                point_seq = (left_top, right_top, right_bottom, left_bottom)
    return max_area



#根据4条线，求4个交点
def get_corner_by_four_lines(vert_lines, hori_lines):
    all_points = []
    for i in range(2):
        for j in range(2):
            point = get_intersection_points(vert_lines[i], hori_lines[j])
            all_points.append([point[1], point[0]])

    return all_points

def get_approx_area(p1, p2, p3, p4):
    top_line = np.abs(
        p1[1] - p3[1]
    )
    bottem_line = np.abs(
        p2[1] - p4[1]
    )
    left_line = np.abs(
        p1[0] - p2[0]
    )
    right_line = np.abs(
        p3[0] - p4[0]
    )
    return (top_line + bottem_line) * (left_line + right_line)
#通过三点直接求面积
def calc_area2(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1,p2,p3
    return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

def is_new_line(theta, rho, valid_data, exist_num):
    # 保证检测到2条垂直的线，两条水平的线
    # vertical_line_num = np.abs(valid_data[:exist_num, 0] - 1.5) > 0.5
    # vertical_line_num = np.sum(vertical_line_num)
    # if vertical_line_num >= 2 and np.abs(theta - 1.5) > 0.5:
    #     return False
    # hori_line_num = np.abs(valid_data[:exist_num, 0] - 1.5) <= 0.5
    # hori_line_num = np.sum(hori_line_num)
    # if hori_line_num >= 2 and np.abs(theta - 1.5) <= 0.5:
    #     return False

    for i in range(exist_num):
        theta = 0 if theta - 3.1 > 0 else theta  # 角度3.1...和零度是一样的
        if theta - valid_data[i][0] < 0.2 and np.square(np.abs(rho) - np.abs(valid_data[i][1])) < 1000:
            # 角度相近 & rho相近
            return False
    return True


def get_intersection_points(line1, line2):
    """
    由极坐标表示的line1, line2,求出角点(矩阵求解方程)
    :param line1: [theta1, rho1]
    :param line2: [theta2,rho2]
    :return: row, col
    """
    rho_mat = np.array(
        [line1[1], line2[1]]
    )
    theta_mat = np.array(

        [[np.cos(line1[0]), np.sin(line1[0])],
         [np.cos(line2[0]), np.sin(line2[0])]]
    )
    inv_theta_mat = np.linalg.inv(theta_mat)
    result = np.matmul(inv_theta_mat, rho_mat).astype(np.int32)
    return result.astype(np.int32)  # 由于是坐标,需要改成int


def harries(gray):
    # corners = cv2.cornerHarris(img, 2, 3, 0.04)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(gray, (x, y), 3, 255, -1)
    return gray


if __name__ == "__main__":
    path = "./data/000026.jpg"
    # path = './data/000872.jpg'
    # path = './data/001201.jpg'
    # path = './data/001402.jpg'
    # path = './data/001552.jpg'
    path = "./data/1.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, (504, 738))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_img, point_seq = detect_corners(gray_img)
    print(np.array(point_seq))
    # plt.imshow(detect_img)
    # plt.show()
    # cv2.imshow('dst', detect_img)
    # cv2.waitKey(0)
