#!/usr/bin/python

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

matplotlib.rcParams.update({'figure.max_open_warning':0})

def showImage(im, boxes, cls):

    classToColor = ['', 'red', 'magenta', 'blue', 'yellow']

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    for i in xrange(boxes.shape[0]):
            bbox = boxes[i]
            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor= classToColor[cls[i]], linewidth=2.0)
                )



            ax.text(bbox[0], bbox[1] - 2,
                    '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

def tattooShowBox(image_set):
    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    gt_roidb = imdb.gt_roidb()

    for i in xrange(num_images):
        if i % 40 == 0 or True:

            im_name = imdb.image_path_at(i)

            print(im_name.strip().split('/')[-1])

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

            print(i)

            showImage(im, bbox, cls)
            plt.savefig( im_name.split('/')[-1] + 'Train.png',
                    bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    tattooShowBox('psdbFourParts_2015_test')
    
