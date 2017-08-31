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

def showImage(im, boxes, gt_boxes):

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)



    gt = gt_boxes
    det = boxes[1]
    det = det[det[:,-1] >= 0.79847407]


    num_gt = len(gt_boxes)
    num_det = det.shape[0]

    '''
    print(gt)
    print(det)
    print(num_gt)
    print(num_det)
    exit(0)
    '''


    gt_hit_mask = np.zeros(num_gt, dtype=bool) 
    det_hit_mask = np.zeros(num_det, dtype=bool)

    iou_thresh = 0.5

    for i in xrange(num_det):
        max_iou = -np.inf
        max_idx = -1
        det_bbox = det[i, :4]

        for j in xrange(num_gt):
            if gt_hit_mask[j]:
                continue

            gt_bbox = gt[j]

            x1 = max(det_bbox[0], gt_bbox[0])
            y1 = max(det_bbox[1], gt_bbox[1])
            x2 = min(det_bbox[2], gt_bbox[2])
            y2 = min(det_bbox[3], gt_bbox[3])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if w > 0 and h > 0:
                s1 = (det_bbox[2] - det_bbox[0] + 1) * \
                    (det_bbox[3] - det_bbox[1] + 1)

                s2 = (gt_bbox[2] - gt_bbox[0] + 1) * \
                    (gt_bbox[3] - gt_bbox[1] + 1)
                
                s3 = w * h
                iou = s3 / (s2 + s1 - s3)

                if iou >= iou_thresh and iou > max_iou:
                    max_iou = iou
                    max_idx = j

        if max_idx >= 0:
            gt_hit_mask[max_idx] = True
            det_hit_mask[i] = True


    print(gt_hit_mask)        
    print(det_hit_mask)


    for i in xrange(num_gt):
        if not gt_hit_mask[i]:
            bbox = gt[i]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1], fill=False,
                    edgecolor='red', linewidth=1.5)
            )

            ax.text(bbox[0], bbox[1] - 2,
                '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), 
                int(bbox[3] - bbox[1])),
                bbox=dict(facecolor='blue', alpha=0.2),
                fontsize=8, color='white')


    for i in xrange(num_det):
        if not det_hit_mask[i]:
            bbox = det[i]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1], fill=False,
                    edgecolor='yellow', linewidth=1.5)
            )

            ax.text(bbox[0], bbox[1] - 2,
                '{:d}, {:d}, {:.3f}'.format(int(bbox[2] - bbox[0]), 
                int(bbox[3] - bbox[1]), bbox[-1]),
                bbox=dict(facecolor='blue', alpha=0.2),
                fontsize=8, color='white')


    '''
    for i in xrange( 1, boxes.shape[0]):
        for j in xrange( len(boxes[i]) ):
            bbox = boxes[i][j]
            if bbox[-1] >= 0.79847407:
                ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
                )
                ax.text(bbox[0], bbox[1] - 2,
                    '{:d}, {:d}, {:.3f}'.format(int(bbox[2] - bbox[0]), 
                    int(bbox[3] - bbox[1]), bbox[-1]),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')
    '''        

    '''
                ax.text(bbox[0], bbox[1] - 2,
                    '{:.3f}'.format(bbox[-1]),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')
    '''




def showBox(image_set):
    cache_file =  \
        'output/pvanet_full1_ohem_DRoiAlign/detviplV4d2_test/zf_faster_rcnn_iter_100000/detections.pkl'


    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            boxes = cPickle.load(fid)
    else:
        print(cache_file + ' not found')

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    gt_roidb = imdb.gt_roidb()
    boxes = np.array(boxes)

    for i in xrange(num_images):
#        if random.randint(1, 100) < 2:
        if i % 10 == 0:
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            gt_boxes = gt_roidb[i]['boxes']
    
            print(i)
            showImage(im, boxes[:, i], gt_boxes)
            plt.savefig(str(i)+'RoiAlign.jpg',bbox_inches='tight', pad_inches=0)
#    plt.show()




if __name__ == '__main__':
    showBox('detviplV4d2_2016_test')
    
