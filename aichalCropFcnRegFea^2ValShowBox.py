#!/usr/bin/env python


import json
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random


def compute_oks(kps, gt_boxes, gt_kps):

    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    kps_count = len(kps)
    gt_count = len(gt_kps)
    oks = np.zeros((gt_count, kps_count))

    for i in range(gt_count):
        gt_kp = np.reshape(gt_kps[i], (14, 3))
        visible = gt_kp[:, 2] == 1
        gt_box = gt_boxes[i]
        scale = np.float32((gt_box[3]-gt_box[1])*(gt_box[2]-gt_box[0]))

        if np.sum(visible) == 0:
            oks[i, :] = 0
            continue
        for j in range(kps_count):
            kp = np.reshape(kps[j], (14, 3))
            dis = np.sum((gt_kp[visible, :2] - kp[visible, :2]) ** 2, axis=1)
            oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
        
    oks = np.max(oks, axis=0)

    return oks




def showImage(im, kps, gt_boxes, gt_kps):

    oks = compute_oks(kps, gt_boxes, gt_kps)

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

    for i, kp in enumerate(kps):

        for j in range(14):
            if kp[j * 3 + 2] == 3:
                continue
            x, y, z = kp[j * 3 : (j + 1) * 3]
            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[i], 
                          linewidth=2.0)
                )
            ax.text(x, y - 2, '{:3f}'.format(oks[i]), 
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

        for l in line:
            i0 = l[0] - 1
            p0 = kp[i0 * 3 : (i0 + 1) * 3] 

            i1 = l[1] - 1
            p1 = kp[i1 * 3 : (i1 + 1) * 3]

            if p0[2] == 3 or p1[2] == 3:
                continue
            
            ax.add_patch(
                    plt.Arrow(p0[0], p0[1], 
                    float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                    color = c[i])
                    )




def showBox():
    with open('pred_fcn_reg_fea^2_val_0.1_0.5.json', 'r') as f:
        data = json.load(f)

    with open('data/aichal/val.json', 'r') as fG:
        dataGt = json.load(fG)

    for i, line in enumerate(data):
#        if i % 1000 == 0:
            imname = line['image_id']
            im = cv2.imread(os.path.join('data/aichal', 'val', imname + '.jpg'))
            kps = line['keypoint_annotations'].values()

            gt = dataGt[i]
            gt_boxes = []
            gt_kps = []
            for key in gt['keypoint_annotations']:
                gt_boxes.append(gt['human_annotations'][key])
                gt_kps.append(gt['keypoint_annotations'][key])

            showImage(im, kps, gt_boxes, gt_kps)
            plt.savefig(str(i)+'pvafcnregfea^2', bbox_inches='tight', pad_inches=0)

            if i == 10:
                break

if __name__ == '__main__':
    showBox()


