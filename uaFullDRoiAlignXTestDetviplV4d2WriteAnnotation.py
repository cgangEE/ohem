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
            if bbox[-1] >= 0.8:
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

def checkPath(annFileName):
    pathes = annFileName.split('/')[:-1]
    path = ''
    for p in pathes:
        path = os.path.join(path, p)
        if not os.path.exists(path):
            os.makedirs(path)

def writeBoxes(annFileName, boxes):
    checkPath(annFileName)

    fOut = open(annFileName, 'w')
    boxes = boxes[1]
    for b in boxes:

        if b[-1] > 0.8:
            x1, y1, x2, y2, p = b
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            fOut.write(str(x1) + ' ' + str(y1) + ' ' +
                    str(w) + ' ' + str(h) + ' 2' + '\n')
    fOut.close()

count = 0

def writeAnnotation(image_set, subset):
    global count 

    cache_file =  \
        'output/pvanet_full1_ohem_DRoiAlignX_testCar/detviplV4d2_' + subset + '/zf_faster_rcnn_iter_100000_inference/detections.pkl'


    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)
    else:
        print(cache_file + ' not found')

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    boxes = np.array(boxes)


    for i in xrange(num_images):
        im_name = imdb.image_path_at(i)
        ann_name = os.path.join('detviplV4d2', 'annotation_car', 
                    im_name.strip().split('images/')[-1][:-3] + 'txt')
        writeBoxes(ann_name, boxes[:, i])
        count += 1

        '''
        im = cv2.imread(im_name)
        showImage(im, boxes[:, i])
        plt.savefig(str(i)+'RoiAlign.jpg',bbox_inches='tight', pad_inches=0)
        '''

if __name__ == '__main__':
    print(count)
    writeAnnotation('detviplV4d2_2016_train', 'train')
    print(count)
    writeAnnotation('detviplV4d2_2016_test', 'test')

    print(count)

    
