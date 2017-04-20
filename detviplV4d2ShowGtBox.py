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
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    classToColor = ['', 'red', 'magenta', 'blue', 'yellow']
    for i in xrange(boxes.shape[0]):
            bbox = boxes[i]
            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor= classToColor[cls[i]], linewidth=2.0)
                )



def tattooShowBox(image_set):
    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    gt_roidb = imdb.gt_roidb()

    for i in xrange(num_images):
#        if random.randint(1, 100) < 2:
#        if i <= 10:
        if i % 50 == 0:
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            
            bbox = gt_roidb[i]['boxes']
            print("")
            print(i)
            cls = gt_roidb[i]['gt_classes']
            print(cls)
            print(imdb.image_path_at(i))
            showImage(im, bbox, cls)

            plt.savefig(str(i) + 'Train')
#    plt.show()



if __name__ == '__main__':
    tattooShowBox('detviplV4d2_2016_test')
    
