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
import json

matplotlib.rcParams.update({'figure.max_open_warning':0})


def showImage(im, boxes, keypoints):
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
                          edgecolor= 'red', linewidth=2.0)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), 
                    int(bbox[3] - bbox[1])),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

            keypoint = keypoints[i]
            for j in range(14):
                x, y = keypoint[j * 2 : (j + 1) * 2]

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


def plotHistByList(prob, title):
    plt.figure()
    plt.hist(prob, bins=30)
    plt.xlabel(title)
    plt.ylabel('#boxes')
    plt.savefig('{}.png'.format(title))


def plotHist(all_rois, all_scores, all_kps):

    prob_rois = []
    for scores in all_scores:
        for score in scores:
            prob_rois.append(score)

    prob_kps = []
    for kps in all_kps:
        for kp in kps:
            prob = sum(kp[2::3]) / 14
            prob_kps.append(prob)

    plotHistByList(prob_rois, 'boxes-prob')
    plotHistByList(prob_kps, 'kps-prob')


def nms(kps, kps_scores, nms_thresh):



def writeJson(image_set, kp_thresh, box_thresh, nms_thresh):

    cache_file =  \
        'output/kpClusterFcn/aichal_val/\
        zf_faster_rcnn_iter_100000_inference/detections.pkl'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            results = cPickle.load(fid)
    else:
        print('ft')


    all_rois = results['all_rois']
    all_scores = results['all_scores']
    all_kps = results['all_kps']


#    plotHist(all_rois, all_scores, all_kps)


    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)


    for i in xrange(num_images):

        rois = all_rois[i]
        scores = all_scores[i];
        kps = all_kps[i]
        kps_scores = [sum(kp[2::3]) / 14 for kp in kps]

        idx = np.where(np.logical_and(
                np.array(scores) > box_thresh, 
                np.array(kps_scores) > kp_thresh))[0]
    
        kps = np.array(kps)[idx]
        kps_scores = np.array(kps_scores)[idx]

        idx = np.argsort(kps_scores)[::-1]
        kps = kps[idx]
        kps_scores = kps_scores[idx]

        keep = nms(kps, kps_scores, nms_thresh)
        print(kps_scores)
        exit(0)


if __name__ == '__main__':
    for kp_thresh in [0.1]:
        for box_thresh in [0.5]:
            for nms_thresh in [0.3]:
                writeJson('aichal_2017_val', kp_thresh, box_thresh, nms_thresh)
    
