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

def showImage(im, boxes, keypoints, cls):
    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
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
                    '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), 
                    int(bbox[3] - bbox[1])),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

            keypoint = keypoints[i]
            for j in range(14):
                x, y, c = keypoint[j * 3 : (j + 1) * 3]
                if c != 3:
                    r = (j % 3) * 0.333
                    g = ((j / 3) % 3) * 0.333
                    b = (j / 3 / 3) * 0.333

                    ax.add_patch(
                            plt.Circle((x, y), 10,
                                  fill=True,
                                  color=(r, g, b), 
                                  edgecolor = (r, g, b), 
                                  linewidth=2.0)
                        )



def tattooShowBox(image_set):
    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    gt_roidb = imdb.gt_roidb()

    for i in xrange(num_images):
        if i % 100 == 0:
            bbox = gt_roidb[i]['boxes']
            keypoints = gt_roidb[i]['keypoints']
            cls = gt_roidb[i]['gt_classes']

            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)

            showImage(im, bbox, keypoints, cls)
            plt.savefig(str(i), bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    tattooShowBox('aichal_2017_train')
    
