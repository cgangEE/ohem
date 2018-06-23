#!/usr/bin/python

import matplotlib 
matplotlib.use('Agg')
import tools._init_paths
import cPickle
import os
from fast_rcnn.config import cfg
import numpy as np
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import cv2
import random

matplotlib.rcParams.update({'figure.max_open_warning':0})

def showImage(im, boxes, cls):

    classToColor = ['', 'red', 'magenta', 'blue', 'yellow']


    for i in xrange(boxes.shape[0]):
        bbox = boxes[i]
        part = im[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        cv2.imwrite(str(i) + '.png', part)

        '''
        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor= classToColor[cls[i]], linewidth=3.0)
            )
        '''

def tattooShowBox(image_set):
    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    gt_roidb = imdb.gt_roidb()

    for i in xrange(num_images):

        im_name = imdb.image_path_at(i)

        if im_name.strip().split('/')[-1] != 's18177.jpg':
            continue

        im = cv2.imread(im_name)
        
        bbox = gt_roidb[i]['boxes']
        head = gt_roidb[i]['head']
        head_shoulder = gt_roidb[i]['head_shoulder']
        upper_body = gt_roidb[i]['upper_body']

        cls = gt_roidb[i]['gt_classes']


        bbox = np.vstack( (bbox, head, head_shoulder, upper_body) )
        cls = np.hstack( \
                (cls, np.ones(cls.shape, dtype=cls.dtype) * 2, 
                       np.ones(cls.shape, dtype=cls.dtype) * 3, 
                       np.ones(cls.shape, dtype=cls.dtype) * 4))


        showImage(im, bbox, cls)



if __name__ == '__main__':
    tattooShowBox('psdbFourParts_2015_test')
    
