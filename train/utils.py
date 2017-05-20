#coding=utf-8
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint
import os

def visualize_body_part(img_bgr,keypoints):
    """
    可视化身体部件
    :param img_bgr: ndarray 存储RGB图像，并且通道顺序位BGR 
    :param keypoints: ndarray 人体关键点标注信息
    :return: 
    """
    # plt.imshow(img_bgr[:,:,(2,1,0)])
    # plt.show()
    # len(colors)为18，但COCO中只有17个关键点，是因为将neck标注加了进去
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    canvas = None
    oriImg = None
    if(type(img_bgr) == str):
        canvas = cv.imread(img_bgr)  # B,G,R order
        oriImg = canvas
    else:
        oriImg = img_bgr
        canvas = img_bgr

    # canvas = canvas.copy()
    # cv.circle(canvas, (100,100), 40, colors[1], thickness=-1)
    # cv.imshow('image', oriImg)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    cmap = matplotlib.cm.get_cmap('hsv')
    if(len(keypoints.shape) == 2):
        all_people = keypoints.reshape(1,keypoints.shape[0],keypoints.shape[1])
    else:
        all_people = keypoints
    all_people = all_people.tolist()
    # pprint.pprint(all_people)
    for i in range(len(all_people)):
        for j in range(18):
            # rgba = np.array(cmap(1 - j / 18. - 1. / 36))
            # rgba[0:3] *= 255
            canvas = canvas.copy()
            # cv.circle(canvas, (100,100), 4, colors[i], thickness=-1)
            if(len(all_people[i][j]) == 3 and all_people[i][j][2] == 2):
                continue
            center = (int(all_people[i][j][0]),int(all_people[i][j][1]))
            cv.circle(canvas,center, 4, colors[j], thickness=-1)
    """
    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    """
    to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()

def visualize_img(img):
    if(len(img.shape) == 3):
        img = img[:,:,(2,1,0)]
    plt.imshow(img)
    plt.show()

def mkdir(name):
    if(os.path.exists(name) == False):
        os.makedirs(name)