#!/usr/bin/env python

import tools._init_paths
import cPickle
import os
from fast_rcnn.config import cfg
import numpy as np
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import matplotlib
import cv2
import random
import pylab as pl
import PIL

matplotlib.rcParams.update({'figure.max_open_warning':0})

def getBox(image_set):

    filename = 'myBox.pkl'
    min_size = 640
    max_size = 1000


    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            myBox = cPickle.load(f)
    else:
        imdb = get_imdb(image_set)
        num_images = len(imdb.image_index)

        gt_roidb = imdb.gt_roidb()
        myBox = []    

        for i in xrange(num_images):
            imname = imdb.image_path_at(i)
            size = PIL.Image.open(imname).size
            im_size_min = min(size[0], size[1])
            im_size_max = max(size[0], size[1])

            im_scale = float(min_size) / float(im_size_min)
            if im_size_max * im_scale > max_size:
                im_scale = float(max_size) / float(im_size_max)
            
            bbox = gt_roidb[i]['boxes']
            for box in bbox:
                b = np.array([box[2] - box[0], box[3] - box[1]]) * im_scale
                myBox.append(list(b))

        with open(filename, 'wb') as f:
            cPickle.dump(myBox, f, cPickle.HIGHEST_PROTOCOL)

    return myBox

def getIoU(box, center):
    box = map(float, box)
    center = map(float, center)
    w = min(box[0], center[0]) * 1.0
    h = min(box[1], center[1]) * 1.0
    if w * h == 0:
        return 0.0
    return w * h / (box[0] * box[1] + center[0] * center[1] - w * h)

def getAveIoU(base_size, scale, ratio, myBox, n):
    center = []
    for r in ratio:
        box = [float(base_size) / np.sqrt(r), 
                float(base_size) * np.sqrt(r)]
         
        for s in scale:
            center.append(np.array(box) * s)

    centerNum = len(center)

    aveIoU = 0.0
    for i in range(n):
        maxIoU = 0.0 
        maxIoUIdx = 0
        for j in range(centerNum):
            IoU = getIoU(myBox[i], center[j])
            if IoU > maxIoU:
                maxIoU = IoU
                maxIoUIdx = j
        aveIoU += maxIoU
    

    aveIoU /= n                    

    
    print(center)
    print(len(center))
    print(aveIoU)

    

def clusterBox(image_set):

    myBox = getBox(image_set)
    n = len(myBox)
    print(n)
    
    base_size = 8
    scale = [3, 6, 9, 16, 32]
    ratio = [1.0, 2.0, 3.0, 4.0, 5.0]
    getAveIoU(base_size, scale, ratio, myBox, n)


            
        

    IoUList = []

    for centerNum in range(1, 26):
        center = []
        for j in range(centerNum):
            center.append(myBox[random.randint(0, n-1)])

        preIoU = 0.0
        for times in range(100):
            centerToBoxIdx = []
            
            for j in range(centerNum):
                centerToBoxIdx.append([])
            
            aveIoU = 0.0
            for i in range(n):
                maxIoU = 0.0 
                maxIoUIdx = 0
                for j in range(centerNum):
                    IoU = getIoU(myBox[i], center[j])
                    if IoU > maxIoU:
                        maxIoU = IoU
                        maxIoUIdx = j
                centerToBoxIdx[maxIoUIdx].append(i)
                aveIoU += maxIoU

            aveIoU /= n                    
            print(centerNum, times, aveIoU)
            for i in range(centerNum):
                print(center[i])
            

            for j in range(centerNum):
                m = len(centerToBoxIdx[j])
                center[j] = [0.0, 0.0]
                for idx in centerToBoxIdx[j]: 
                    center[j][0] += myBox[idx][0] * 1.0 / m
                    center[j][1] += myBox[idx][1] * 1.0 / m



            if aveIoU - preIoU <= 0.006:
                IoUList.append(aveIoU)
                break
            preIoU = aveIoU



    print(len(IoUList))
    pl.plot(range(1, len(IoUList) + 1), IoUList, 'o')
    pl.savefig('iouList_0.006.png')

if __name__ == '__main__':
    clusterBox('aichal_2017_val')
    
