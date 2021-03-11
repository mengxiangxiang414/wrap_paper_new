import cv2
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from utils import Conv2d
from lines_cross import *
from clear_grad_img import clear_grad_function
from find_out_line import find_final_points
from get_LSD_lines import get_LSD

'''
可调参数
@get_grad_img: grad阈值
'''
top_k = 8
low_thd = 100
up_thd = 200
theta_times = 2
rho_times = 2
corner_area = 20
vote_rate = 0.5
max_min_ratio = 1.8#最长边比最短边


def houghLines(grad_img):
    """
    从梯度图进行hough变换，检测直线
    :param grad_img: 梯度图
    :return: 检测出来的直线的极坐标表示
    """
    # --------------------------------------投票------------------------------------------
    rho_max = math.sqrt(grad_img.shape[0] * grad_img.shape[0] + grad_img.shape[1] * grad_img.shape[1])
    #rho_max = max(grad_img.shape)
    time1 = time.time()
    if theta_times != 0:
        m = 180*theta_times//2
    if rho_times != 0:
        n = int(max(grad_img.shape)*rho_times/2)
    #m, n = 180, int(rho_max)
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
    #print("pre:{}".format(time2 - time1))
    # ----------------------------------过滤 1： 选出不同的直线-------------------------------
    # 取出top_k条不同的直线
    #top_k = top_k
    #print("top_k=",top_k)
    top_vert = 0
    top_hori = 0
    # unravel_index: https://www.jianshu.com/p/a7e19847bd39 -> 将一维index转换到(m,n)维上
    argmax_ind = np.argsort(-vote_table.ravel(), )
    argmax_ind = np.unravel_index(argmax_ind, (m, n))
    argmax_ind = np.dstack(argmax_ind)
    time3 = time.time()
    #print("top:{}".format(time3 - time2))
    argmax_ind = argmax_ind[0, :, :]
    valid_lines = np.zeros((top_k, 2))
    exist_num = 0

    #找主方向
    main_theta = 0
    for i in range(10):
        main_theta_ind,main_rho_ind = tuple(argmax_ind[i])
        temp_main_theta = theta_range[main_theta_ind]
        if temp_main_theta > 1.5:
            temp_main_theta2 = 3.141 - temp_main_theta
            if temp_main_theta2 < 0.3 or temp_main_theta2 > 1.2:
                main_theta = temp_main_theta
                break
        else:
            if temp_main_theta < 0.3 or temp_main_theta > 1.2:
                main_theta = temp_main_theta
                break

    # 在中心点的上下左右各设置一个数组，来保存合格的直线
    left_lines = []
    top_lines = []
    right_lines = []
    bottom_lines = []

    left_left_lines = []
    top_top_lines = []
    right_right_lines = []
    bottom_bottom_lines = []

    left_lines_offset = []
    top_lines_offset = []
    right_lines_offset = []
    bottom_lines_offset = []

    all_lines_stop = [1,1,1,1]

    time_ccc = time.time()
    for i in range(0, 500):
        row_ind, col_ind = tuple(argmax_ind[i])
        theta = theta_range[row_ind]
        rho = rho_range[col_ind]
        if theta==0 and rho==0:
            continue
        delta_theta = abs(theta-main_theta)
        if delta_theta>1.5:
            delta_theta = 3.141-delta_theta
        if delta_theta>0.25 and delta_theta<1.25:
            continue

        if is_new_line2(theta, rho, valid_lines, exist_num, grad_img.shape,
                       left_lines,top_lines,right_lines,bottom_lines, vote_table[row_ind,col_ind], all_lines_stop,
                        left_left_lines,top_top_lines,right_right_lines,bottom_bottom_lines,
                        left_lines_offset,top_lines_offset,right_lines_offset,bottom_lines_offset):
        # if is_new_line(theta, rho, valid_lines, exist_num, grad_img.shape,
        #                    left_lines, top_lines, right_lines, bottom_lines, vote_table[row_ind, col_ind],
        #                    all_lines_stop):
        #if True:
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
            # if max(all_lines_stop)==0:
            #     break
            # if i>200 and len(left_lines)>0 and len(top_lines)and len(right_lines) and len(bottom_lines):
            #     break

    time_cccd = time.time()
    #print("cccddd{}".format(time_cccd-time_ccc))
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
    if (-0.1*w < x and x < 1.1*w) and (-0.1*h < y and y < 1.1*h):
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


def detect_corners(gray_img, top_lines, low_threshold, up_threshold, theta, rho, corner_area_len, vote_r):
    global top_k,low_thd,up_thd, theta_times, rho_times, corner_area, vote_rate
    top_k = top_lines
    low_thd = low_threshold
    up_thd = up_threshold
    theta_times = theta
    rho_times = rho
    corner_area = corner_area_len
    vote_rate = vote_r

    time1 = time.time()

    time2 = time.time()
    h,w = gray_img.shape

    grad_img = get_LSD(gray_img)
    # cv2.imshow('LSD_lines', grad_img)
    #grad_img = cv2.resize(grad_img,(w,h))
    #cv2.imshow('half_LSD_line',grad_img)

    time3 = time.time()
    print("LSD_lines:{}".format(time3 - time2))
    polar_lines = houghLines(grad_img)

    time4 = time.time()
    print("houghLines:{}".format(time4 - time3))


    # for i in range(polar_lines.shape[0]):
    #     theta, rho = tuple(polar_lines[i])
    #     # print(theta, rho)
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     #逐条显示画出来的线
    #     cv2.line(grad_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # cv2.imshow('after', grad_img)

    #cv2.waitKey(0)
    time5 = time.time()
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
    vert_lines_actual = []
    hori_lines_actual = []
    for i in range(vert_lines.shape[0]):
        if not(vert_lines[i,0] == 0 and vert_lines[i,1] == 0):
            vert_lines_actual.append(vert_lines[i])
    for j in range(hori_lines.shape[0]):
        if not(hori_lines[j,0] == 0 and hori_lines[j,1] == 0):
            hori_lines_actual.append(hori_lines[j])

    vert_lines_actual = np.array(vert_lines_actual)
    hori_lines_actual = np.array(hori_lines_actual)
    num_vert_lines = vert_lines_actual.shape[0]
    num_hori_lines = hori_lines_actual.shape[0]
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
            temp_vert_lines = vert_lines_actual[couples_vert[i]]
            temp_hori_lines = hori_lines_actual[couples_hori[j]]
            area_table[i][j] = get_area_by_four_lines(temp_vert_lines,temp_hori_lines,gray_img.shape[:2])

    #求出最大的面积，以及其所对应的直线的index
    m,n = np.unravel_index(np.argmax(area_table),(num_vert_couples,num_hori_couples))
    final_vert_lines = vert_lines_actual[couples_vert[m]]
    final_hori_lines = hori_lines_actual[couples_hori[n]]
    points = get_corner_by_four_lines(final_vert_lines, final_hori_lines)
    points = np.array(points).reshape(2, 2, 2)

    time6 = time.time()
    #print("area:{}".format(time6 - time5))

    left_top = points[0][0]
    left_bottom = points[0][1]
    right_top = points[1][0]
    right_bottom = points[1][1]

    point_seq = (left_top, right_top, right_bottom, left_bottom)


    return grad_img, point_seq

#将线段写入txt
def write_lines(lines):
    f = open("g:/lines.txt", "w")
    for i in range(len(lines)):
        for j in range(4):
            f.write("{} ".format(lines[i][j]))
        f.write("\n")
    f.close()

#根据四条直线，计算所围成的面积
def get_area_by_four_lines(vert_lines,hori_lines, img_shape):
    num_vert_lines = 2
    num_hori_lines = 2

    points = get_corner_by_four_lines(vert_lines,hori_lines)

    #判断四个角点是否出界
    out_nums = [0,0,0,0]
    temp_i = 0
    for i in range(4):
        out_bd = out_boundary(points[i],img_shape)
        if out_bd:
            temp_i = i
            out_nums[i] = 1
    # if sum(out_nums) == 2:
    #     if (out_nums[0]==1 and out_nums[2]==1) or (out_nums[1]==1 and out_nums[3]==1):
    #         pass
    #     else:
    #         return 0
    if sum(out_nums) > 2:        #若有3个及以上角点在界外，则直接返回0
        return 0
    #如果只有一个角出界，则判断是否离边界太远
    if out_nums==1:
        out_bd_too_far = out_boundary_too_far(points[temp_i],img_shape)
        if out_bd_too_far:
            return 0#若有两个角点在界外，则直接返回0

    #判断四个交点是否交叉
    p0 = MyPoint(points[0][1],points[0][0])
    p1 = MyPoint(points[1][1],points[1][0])
    p2 = MyPoint(points[2][1], points[2][0])
    p3 = MyPoint(points[3][1], points[3][0])

    d = IsIntersec(p0, p1, p2, p3)
    if d==1:
        return 0

    #若

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
                #point_seq = (left_top, right_top, right_bottom, left_bottom)
    # 若最长边/最短边>2,则返回0
    edge_len = []
    edge_len.append(np.sqrt((left_top[1] - right_top[1]) ** 2 + (left_top[0] - left_top[0]) ** 2))
    edge_len.append(np.sqrt((right_top[1] - right_bottom[1]) ** 2 + (right_top[0] - right_bottom[0]) ** 2))
    edge_len.append(np.sqrt((right_bottom[1] - left_bottom[1]) ** 2 + (right_bottom[0] - left_bottom[0]) ** 2))
    edge_len.append(np.sqrt((left_bottom[1] - left_top[1]) ** 2 + (left_bottom[0] - left_top[0]) ** 2))
    max_edge = max(edge_len)
    min_edge = min(edge_len)
    if max_edge/min_edge>max_min_ratio:
        return 0
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

def is_new_line(theta, rho, valid_data, exist_num, img_shape, left_lines,top_lines,right_lines,bottom_lines, line_vote, all_lines_stop):

    is_vert = np.abs(theta - 1.5) > 0.5

    centerX = img_shape[1]//2
    centerY = img_shape[0]//2
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    dis =getDis(centerX,centerY,x1,y1,x2,y2)
    min_dis = min(img_shape)
    if dis < 0.1*min_dis:
        return False

    for i in range(exist_num):
        theta = 0 if theta - 3.1 > 0 else theta  # 角度3.1...和零度是一样的
        if theta - valid_data[i][0] < 0.2 and np.square(np.abs(rho) - np.abs(valid_data[i][1])) < 100:
            # 角度相近 & rho相近
            return False
        #求两直线的交点，若角点在图内，且夹角很小，则丢弃

        delta = abs(theta - valid_data[i][0])
        if delta>1.5:
            delta = 3.14-delta
        if delta > 0.001 and delta <1.2 :
            cross_point = get_intersection_points(np.array(valid_data[i]), [theta, rho])
            if cross_point[0]>-0.5*img_shape[0] and \
                cross_point[0]<1.5*img_shape[0] and \
                cross_point[1]>-0.5*img_shape[1] and \
                cross_point[1]<1.5*img_shape[1]:
                return False

    return True

def is_new_line2(theta, rho, valid_data, exist_num, img_shape, left_lines,top_lines,right_lines,bottom_lines, line_vote, all_lines_stop,
                 left_left_lines,top_top_lines,right_right_lines,bottom_bottom_lines,
                 left_lines_offset,top_lines_offset,right_lines_offset,bottom_lines_offset):

    is_vert = np.abs(theta - 1.5) > 0.5

    centerX = img_shape[1]//2
    centerY = img_shape[0]//2
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    dis =getDis(centerX,centerY,x1,y1,x2,y2)
    min_dis = min(img_shape)
    if dis < 0.2*min_dis:
        return False

    for i in range(exist_num):
        theta = 0 if theta - 3.1 > 0 else theta  # 角度3.1...和零度是一样的
        if theta - valid_data[i][0] < 0.2 and np.square(np.abs(rho) - np.abs(valid_data[i][1])) < 100:
            # 角度相近 & rho相近
            return False
        #求两直线的交点，若角点在图内，且夹角很小，则丢弃

        delta = abs(theta - valid_data[i][0])
        if delta>1.5:
            delta = 3.14-delta
        if delta > 0.001 and delta <1.2 :
            cross_point = get_intersection_points(np.array(valid_data[i]), [theta, rho])
            if cross_point[0]>-0.5*img_shape[0] and \
                cross_point[0]<1.5*img_shape[0] and \
                cross_point[1]>-0.5*img_shape[1] and \
                cross_point[1]<1.5*img_shape[1]:
                return False
    # 判断直线在中心点的上、下、左、又
    if is_vert:
        x3 = (x2 - x1) / (y2 - y1) * (centerY - y1) + x1
        if x3 > centerX:
            if len(right_lines) >= (top_k // 4):
                return False
            if right_lines==[]:
                right_lines.append(line_vote)
                right_lines_offset.append(x3)
            else:
                max_vote = max(right_lines)
                max_idx = right_lines.index(max_vote)
                if line_vote < max_vote*vote_rate:
                    all_lines_stop[0] = 0
                    if x3>right_lines_offset[max_idx]:
                        right_right_lines.append([theta,rho])
                    return False
                else:
                    right_lines.append(line_vote)
                    right_lines_offset.append(x3)
        else:
            if len(left_lines) >= (top_k // 4):
                max_vote = max(left_lines)
                max_idx = left_lines.index(max_vote)

                if x3 < left_lines_offset[max_idx]:
                    left_left_lines.append([theta, rho])
                return False
            if left_lines == []:
                left_lines.append(line_vote)
                left_lines_offset.append(x3)
            else:
                max_vote = max(left_lines)
                max_idx = left_lines.index(max_vote)
                if line_vote < max_vote * vote_rate:
                    all_lines_stop[1] = 0
                    # if x3<left_lines_offset[max_idx]:
                    #     left_left_lines.append([theta,rho])
                    return False
                else:
                    left_lines.append(line_vote)
                    left_lines_offset.append(x3)
    else:
        y3 = (y2 - y1) / (x2 - x1) * (centerX - x1) + y1
        if y3 > centerY:
            if len(bottom_lines) >= (top_k // 4):
                return False
            if bottom_lines == []:
                bottom_lines.append(line_vote)
                bottom_lines_offset.append(y3)
            else:
                max_vote = max(bottom_lines)
                max_idx = bottom_lines.index(max_vote)
                if line_vote < max_vote * vote_rate:
                    all_lines_stop[2] = 0
                    if y3>bottom_lines_offset[max_idx]:
                        bottom_bottom_lines.append([theta,rho])
                    return False
                else:
                    bottom_lines.append(line_vote)
                    bottom_lines_offset.append(y3)
        else:
            if len(top_lines) >= (top_k // 4):
                return False
            if top_lines == []:
                top_lines.append(line_vote)
                top_lines_offset.append(y3)
            else:
                max_vote = max(top_lines)
                max_idx = top_lines.index(max_vote)
                if line_vote < max_vote * vote_rate:
                    all_lines_stop[3] = 0
                    if y3<top_lines_offset[max_idx]:
                        top_top_lines.append([theta,rho])
                    return False
                else:
                    top_lines.append(line_vote)
                    top_lines_offset.append(y3)
    return True
#点到直线的距离
def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    return dis

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


