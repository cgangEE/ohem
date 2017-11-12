#!/usr/bin/env python


import json
import cPickle
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import random


def showImage(im, keypoints):

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

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    i = 0
    for key in keypoints:
        
        keypoint = keypoints[key]
        for j in range(14):
            if keypoint[j * 3 + 2] == 3:
                continue

            x, y, z = keypoint[j * 3 : (j + 1) * 3]


            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[i], 
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
        i += 1


def showBox():
    with open('pred_cluster_fcn_val_0.4_0.3_0.9.json', 'r') as f:
        data = json.load(f)

    for i, line in enumerate(data):
        if i % 1000 == 0:
            imname = line['image_id']
            im = cv2.imread(os.path.join('data/aichal', 'val', imname + '.jpg'))
            kps = line['keypoint_annotations']
            showImage(im, kps)
            plt.savefig(str(i), bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    showBox()
