#!/usr/bin/env python


import json
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tools._init_paths
from fast_rcnn.config import cfg
from datasets.factory import get_imdb


Delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
    0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
    0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])


def load_annotations(anno_file, return_dict):
    """Convert annotation JSON file."""
    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = 2*np.array(
            [0.01388152, 0.01515228, 0.01057665, 0.01417709, \
            0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
            0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def compute_oks(anno, predict, delta=Delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = anno['keypoint_annos'].keys()[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i, j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = predict.keys()[j]
                predict_keypoints = np.reshape(predict[predict_key], (14, 3))
                dis = np.sum((anno_keypoints[visible, :2] \
                    - predict_keypoints[visible, :2])**2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))
    return oks




def showImage(im, keypoints, gt_kps, kpType):
    oks = compute_oks(gt_kps, keypoints)
    gt_assignment = np.argmax(oks, axis=0)

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
    for i, key in enumerate(keypoints):
        
        keypoint = keypoints[key]

        anno_key = gt_kps['keypoint_annos'].keys()[gt_assignment[i]]
        gt = np.reshape(gt_kps['keypoint_annos'][anno_key], (14, 3))
        for j in range(14):

            if gt[j][2] <= kpType:
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

            if gt[i1,2] <= kpType and gt[i0,2] <= kpType:

                ax.add_patch(
                        plt.Arrow(p0[0], p0[1], 
                        float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                        color = c[i])
                        )



def showBox():
    return_dict = dict()
    return_dict['error'] = None
    return_dict['warning'] = []
    return_dict['score'] = None

    # Load annotation JSON file
    annotations = load_annotations(anno_file='data/aichal/val.json',
                                   return_dict=return_dict)

    with open('pred_fcn_reg_val_0.1_0.5.json', 'r') as f:
        data = json.load(f)

    kpType = 1

    for i, line in enumerate(data):
        if i % 1000 == 0:

            imname = line['image_id']
            im = cv2.imread(os.path.join('data/aichal', 'val', imname + '.jpg'))
            kps = line['keypoint_annotations']
            showImage(im, kps, annotations['annos'][imname], kpType)
            plt.savefig(str(i) + 'kpType' + str(kpType), 
                bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    showBox()
