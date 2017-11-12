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


def showImage(im, rois, kps, imageIdx):

    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i, kp in enumerate(kps):
        
        for j in range(14):
            x, y, z = kp[j * 3 : (j + 1) * 3]
            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[i], 
                          linewidth=2.0)
                )

        for l in line:
            i0 = l[0] - 1
            p0 = kp[i0 * 3 : (i0 + 1) * 3] 

            i1 = l[1] - 1
            p1 = kp[i1 * 3 : (i1 + 1) * 3]

            ax.add_patch(
                    plt.Arrow(p0[0], p0[1], 
                    float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                    color = c[i])
                    )

    plt.savefig(str(imageIdx))



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



def compute_oks(anno, bbox, predict, kp_thresh):
    """Compute oks matrix (size gtN*pN)."""

    delta = 2*np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709, \
            0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803, \
            0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    predict_count = len(predict)
    oks = np.zeros(predict_count)


    anno_keypoints = np.reshape(anno, (14, 3))
    visible = anno_keypoints[:, 2] >= kp_thresh
    scale = np.float32((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))

    # for every predicted human
    for j in range(predict_count):
        predict_keypoints = np.reshape(predict[j], (14, 3))
        dis = np.sum((anno_keypoints[visible, :2] \
            - predict_keypoints[visible, :2])**2, axis=1)
        oks[j] = np.mean(np.exp(-dis/2/delta[visible]**2/(scale+1)))

    return oks



def nms(kps, kps_scores, rois, nms_thresh, kp_thresh):
    """Pure Python NMS baseline."""

    order = kps_scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks = compute_oks(kps[i], rois[i], kps[order[1:]], kp_thresh)
        inds = np.where(oks <= nms_thresh)[0]
        order = order[inds + 1]

    return keep



def writeJson(image_set, results, nms_thresh, kp_thresh, box_thresh):




    all_rois = np.array(results['all_rois'])
    all_scores = np.array(results['all_scores'])
    all_kps = np.array(results['all_kps'])

    '''
    results['all_rois'] = all_rois
    results['all_scores'] = all_scores
    results['all_kps'] = all_kps

    with open ('tmp.pkl', 'wb') as f:
        cPickle.dump(results, f)
    exit(0)
    '''

#    plotHist(all_rois, all_scores, all_kps)

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    data = []

    for i in xrange(all_rois.shape[0]):
        print(i)

        imname = imdb.image_path_at(i)
        im = cv2.imread(imname)

        image = dict()
        imname = imname.split('/')[-1][:-4]
        image['image_id'] = imname
        kp_ann = dict()

        rois = all_rois[i]
        scores = all_scores[i];
        kps = all_kps[i]
        kps_scores = [sum(kp[2::3]) / 14 for kp in kps]

        idx = np.where(np.logical_and(
                scores > box_thresh, 
                kps_scores > kp_thresh))[0]
    
        kps = np.array(kps)[idx]
        kps_scores = np.array(kps_scores)[idx]
        rois = np.array(rois)[idx]


        keep = nms(kps, kps_scores, rois, nms_thresh, kp_thresh)
        rois = rois[keep]
        kps = kps[keep]
        kps_scores = kps_scores[keep]

        for j, kp in enumerate(kps):
            pred_kp = []
            for k in range(14):
                pred_kp += [ kp[k*3], kp[k*3+1], 1.0 ]

            kp_ann['human' + str(j+1)] = map(int, pred_kp)


        image['keypoint_annotations'] = kp_ann
        data.append(image)

        #showImage(im, rois, kps, i)


    with open('pred_cluster_fcn_val_{}_{}_{}.json'.format(
            nms_thresh, kp_thresh, box_thresh),  'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    cache_file =  \
        'output/kpClusterFcn/aichal_val/\
zf_faster_rcnn_iter_100000_inference/detections.pkl'

    with open(cache_file, 'rb') as fid:
        results = cPickle.load(fid)

    for nms_thresh in [0.05, 0.1, 0.2, 0.3, 0.4]:
        for kp_thresh in [0.1, 0.2, 0.3, 0.4]:
            for box_thresh in [0.3, 0.5, 0.7, 0.9]:
                    writeJson('aichal_2017_val', results,
                            nms_thresh, kp_thresh, box_thresh)
    

