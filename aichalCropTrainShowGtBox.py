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


def showImage(im, boxes, keypoints):
    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    thresh = 0.7

    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', \
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    print(boxes.shape)
    for i in xrange(boxes.shape[0]):

            keypoint = keypoints[i]
            for j in range(14):
                if keypoint[j * 3 + 2] == 3:
                    continue

                x, y, z = keypoint[j * 3 : (j + 1) * 3]


                ax.add_patch(
                        plt.Circle((x, y), 3,
                              fill=True,
                              color = c[j], 
                              linewidth=2.0)
                    )
            for l in line:
                i0 = l[0] - 1
                p0 = keypoint[i0 * 3 : (i0 + 1) * 3] 

                i1 = l[1] - 1
                p1 = keypoint[i1 * 3 : (i1 + 1) * 3]


                if p0[2] == 3 or p1[2] == 3:
                    continue
                
                ax.add_patch(
                        plt.Arrow(p0[0], p0[1], 
                        float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                        color = c[i])
                        )


def showBox(image_set):
    imdb = get_imdb(image_set)
    imdb.append_flipped_images()

    num_images = len(imdb.image_index)

    gt_roidb = imdb.roidb

    for i in xrange(num_images):
        if i % 3000 == 0:
            bbox = gt_roidb[i]['boxes']
            keypoints = gt_roidb[i]['keypoints']
            cls = gt_roidb[i]['gt_classes']

            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            
            print(i, gt_roidb[i]['flipped'])                
            if gt_roidb[i]['flipped']:
                im = im[:, ::-1, :]

            showImage(im, bbox, keypoints)
            plt.savefig(str(i) + 'GT', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    showBox('aichalCrop_2017_train')
    
