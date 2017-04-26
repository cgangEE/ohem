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

def showImage(im, boxes):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    for i in xrange( 1, boxes.shape[0]):
        for j in xrange( len(boxes[i]) ):
            bbox = boxes[i][j]
            if bbox[-1] >= 0.7714887:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
                )

                ax.text(bbox[0], bbox[1] - 2,
                    '{:d}, {:d}, {:.3f}'.format(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]), bbox[-1]),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')
                '''
                ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(bbox[-1]),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')
                '''




def tattooShowBox(image_set):
    cache_file =  \
        'output/pvanet_full1_ohem_D/detviplV4d2_test/zf_faster_rcnn_iter_60000/detections.pkl'


    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)
    else:
        print(cache_file + ' not found')

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

#j    gt_roidb = imdb.gt_roidb()
    boxes = np.array(boxes)

    for i in xrange(num_images):
#        if random.randint(1, 100) < 2:
        if i % 40 == 0 :
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            print(i)
            showImage(im, boxes[:, i])
            plt.savefig(str(i)+'.jpg',bbox_inches='tight', pad_inches=0)
#    plt.show()




if __name__ == '__main__':
    tattooShowBox('detviplV4d2_2016_test')
    
