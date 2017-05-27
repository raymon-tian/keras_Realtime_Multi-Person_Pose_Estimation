#coding=utf-8
import matplotlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pprint
import os
from cStringIO import StringIO
import PIL.Image
from IPython.display import Image, display

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


def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def showmap(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

#def checkparam(param):
#    octave = param['octave']
#    starting_range = param['starting_range']
#    ending_range = param['ending_range']
#    assert starting_range <= ending_range, 'starting ratio should <= ending ratio'
#    assert octave >= 1, 'octave should >= 1'
#    return starting_range, ending_range, octave

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad