# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform_kp import clip_boxes, bbox_transform_inv, kp_transform_inv, clip_kps
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from scipy import signal
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from utils.cython_bbox import bbox_vote
import matplotlib.pyplot as plt


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        # Make width and height be multiples of a specified number
        im_scale_x = np.floor(im.shape[1] * im_scale / cfg.TEST.SCALE_MULTIPLE_OF) * cfg.TEST.SCALE_MULTIPLE_OF / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / cfg.TEST.SCALE_MULTIPLE_OF) * cfg.TEST.SCALE_MULTIPLE_OF / im.shape[0]
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y]))
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def showImage(im, box, score, kp, imageIdx, boxIdx):
    print(box)
    print(kp)

    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]

    thresh = 0.5

    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


    ax.add_patch(
            plt.Rectangle((box[0], box[1]),
                  box[2] - box[0],
                  box[3] - box[1], fill=False,
                  edgecolor= 'r', linewidth=2.0)
        )

    prob = sum(kp[2::3]) / 14

    ax.text(box[0], box[1] - 2, '{:.3f} {:.3f}'.format(score, prob),
            bbox=dict(facecolor='blue', alpha=0.2),
            fontsize=8, color='white')
             

    for j in range(14):
        x, y, p = kp[j * 3 : (j + 1) * 3]

        ax.add_patch(
                plt.Circle((x, y), 3,
                      fill=True,
                      color = c[1], 
                      linewidth=2.0))


    for l in line:
        i0 = l[0] - 1
        p0 = kp[i0 * 3 : (i0 + 1) * 3] 

        i1 = l[1] - 1
        p1 = kp[i1 * 3 : (i1 + 1) * 3]
        
        ax.add_patch(
                plt.Arrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], 
                color = c[2])
                )


    plt.savefig('{}_{}_repoolRect.png'.format(imageIdx, boxIdx), 
            bbox_inches='tight', pad_inches=0)
    


def im_detect(net, im, _t, idx):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    _t['im_preproc'].tic()
    blobs, im_scales = _get_blobs(im, None)


    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [np.hstack((im_blob.shape[2], im_blob.shape[3], im_scales[0]))],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    net.blobs['data'].data[...] = blobs['data']
    #forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].data[...] = blobs['im_info']
        #forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        net.blobs['rois'].data[...] = blobs['rois']
        #forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    _t['im_preproc'].toc()

    _t['im_net'].tic()
    blobs_out = net.forward()
    _t['im_net'].toc()


    _t['im_postproc'].tic()


    rois = net.blobs['rois_repool'].data.copy()[:,1:5] / im_scales[0]
    scores = net.blobs['cls_prob'].data.copy()[:,1]

    kpFcn = blobs_out['pred_fcn_prob'].reshape(-1, 28, 192, 192)

    R = cfg.TEST.RADIUS
    R2 = R*2

    H = 192
    W = 192
    f = np.zeros((R * 2 + 1, R * 2 + 1))
    cv2.circle(f, (R, R), R, 1, -1)
   
    kps = []

    for i, box in enumerate(rois):
        kp = []
        x1, y1, x2, y2 = box

        for j in range(14):

            '''
            kpMap = kpFcn[i, j * 2 + 1].copy()
            kpMap = signal.fftconvolve(kpMap, f, mode='same') / (R * R * 3.14)
            
            p = np.max(kpMap)
            y, x = np.unravel_index(np.argmax(kpMap), kpMap.shape)
            '''

            prob = kpFcn[i, j * 2 + 1].copy()

            w, h = prob.shape
            prob = np.cumsum(np.cumsum(prob, axis=0), axis=1)

            prob_ij = np.lib.pad(prob, ((R2, 0), (R2, 0)), 
                    'constant', constant_values=0)[:w,:h]
            prob_i = np.lib.pad(prob, ((R2, 0), (0, 0)), 
                    'constant', constant_values=0)[:w,:]
            prob_j = np.lib.pad(prob, ((0, 0), (R2, 0)), 
                    'constant', constant_values=0)[:,:h]


            prob = prob - prob_i - prob_j + prob_ij


            p = np.max(prob) / (R2 * R2)
            y, x = np.unravel_index(np.argmax(prob), prob.shape)
            x -= R
            y -= R

            x = float(x) / (W - 1) * (x2 - x1) + x1
            y = float(y) / (H - 1) * (y2 - y1) + y1
            kp += [x, y, p]

#        showImage(im, box, scores[i], kp, idx, i)

        kps.append(kp) 

#    if idx == 10:
#        exit(0)

    _t['im_postproc'].toc()

    return rois, scores, kps



def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes



def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    all_rois = []
    all_scores = []
    all_kps = []

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}


    for i in xrange(num_images):

        im = cv2.imread(imdb.image_path_at(i))
        rois, scores, kps = im_detect(net, im, _t, i)



        _t['misc'].tic()
        all_rois.append(rois)
        all_scores.append(scores)
        all_kps.append(kps)
        _t['misc'].toc()




        print 'im_detect: {:d}/{:d}  net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
              .format(i + 1, num_images, _t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                      _t['misc'].average_time)

    results = {"all_rois" : all_rois, 
                "all_scores" : all_scores,
                "all_kps" : all_kps
                }

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(results, f, cPickle.HIGHEST_PROTOCOL)

#    print 'Evaluating detections'
#    imdb.evaluate_detections(all_boxes, output_dir)
