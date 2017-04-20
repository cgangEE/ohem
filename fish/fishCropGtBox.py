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

    classToColor = ['', 'r', 'm', 'b', 'y', 'g', 'c', 'k', 'w']
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

    newImageIdx = 0

    for i in xrange(num_images):
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            
            bbox = gt_roidb[i]['boxes']
            cls = gt_roidb[i]['gt_classes']

            for i, b in enumerate(bbox):
                c = cls[i]

                for x in range(10): 
                    b1 = random.randint(0, b[1])
                    b0 = random.randint(0, b[0])
                     
                    b3 = random.randint(min(b[3], im.shape[0]), im.shape[0])
                    b2 = random.randint(min(b[2], im.shape[1]),  im.shape[1])

                    cropped = im[ b1:b3, b0:b2 ]
#                cropped = im[ b[1]:b[3], b[0]:b[2] ]

                    newImageIdx += 1
                    newImageName = os.path.join('./data/fish/new2', 
                            imdb._classes[c], str(newImageIdx) + '.jpg')
                    print(newImageName)
                    cv2.imwrite(newImageName, cropped)







if __name__ == '__main__':
    tattooShowBox('fish_2016_train')
    
