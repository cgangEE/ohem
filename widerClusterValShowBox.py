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


def checkDir(dirname):
    dirname = dirname.strip().split('/')
    fullname = ''
    for name in dirname:
        fullname = os.path.join(fullname, name)
        if not os.path.isdir(fullname):
            os.mkdir(fullname)


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
            if bbox[-1] >= 0.5:
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




def showBox(image_set):
    cache_file =  \
        'output/wider_cluster/wider_val/zf_faster_rcnn_iter_100000_inference/detections.pkl'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)
    else:
        print(cache_file + ' not found')

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)
    boxes = np.array(boxes)


    for i in xrange(num_images):
        im_name = ['pred'] + imdb.image_path_at(i).strip().split('/')[8:]

        checkDir('/'.join(im_name[:-1]))
        im_name = '/'.join(im_name)[:-3] + 'txt'

        bbox = boxes[1, i]
        fOut = open(im_name, 'w')

        fOut.write(im_name.split('/')[-1][:-4] + '\n')
        fOut.write(str(len(bbox)) + '\n')
        for b in bbox:
            fOut.write('{} {} {} {} {}\n'.format(b[0], b[1], 
                    b[2] - b[0], b[3] - b[1], b[4]))

        '''
        if i % 100 == 0:
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            print(i)
            showImage(im, boxes[:, i])
            plt.savefig(str(i), bbox_inches='tight', pad_inches=0)
        '''            

        fOut.close()




if __name__ == '__main__':
    showBox('wider_2015_val')
    
