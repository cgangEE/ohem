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
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from utils.cython_bbox import bbox_vote
import matplotlib.pyplot as plt
import scipy
import numpy as np
import scipy.stats as st


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


def showImage(im, kp, imageId):

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

    thresh = 0.01

    for j in range(14):
        x, y, p = kp[j * 3 : (j + 1) * 3]
        '''
        if p < thresh:
            continue
        '''            

        ax.add_patch(
                plt.Circle((x, y), 3,
                      fill=True,
                      color = c[1], 
                      linewidth=2.0)
            )

        '''
        ax.text(x, y - 2, '{:.3f}'.format(kp_scores[i, j]),
                bbox=dict(facecolor='blue', alpha=0.2),
                fontsize=8, color='white')
        '''

    for l in line:
        i0 = l[0] - 1
        p0 = kp[i0 * 3 : (i0 + 1) * 3] 

        i1 = l[1] - 1
        p1 = kp[i1 * 3 : (i1 + 1) * 3]

        '''
        if p0[2] < thresh or p1[2] < thresh:
            continue
        '''
        
        ax.add_patch(
                plt.Arrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], 
                color = c[2])
                )

    plt.savefig(str(imageId) + 'regRect' , bbox_inches='tight', pad_inches=0)
    

def gkern(kernlen=25, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.max()
    return kernel


def im_detect(net, im, _t, idx):

    _t['im_preproc'].tic()
    blobs, im_scales = _get_blobs(im, None)


    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))

    # do forward
    net.blobs['data'].data[...] = blobs['data']
    #forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}

    _t['im_preproc'].toc()


    _t['im_net'].tic()
    blobs_out = net.forward()
    _t['im_net'].toc()


    _t['im_postproc'].tic()

    score = net.blobs['rpn_cls_prob'].data.copy()
    reg = net.blobs['upsample/rpn_cls_reg'].data.copy()

    R = cfg.TRAIN.RADIUS 
    h, w = reg[0,0].shape

    coord_x = np.multiply(np.ones((h,1), dtype=np.int), 
            np.arange(w)[np.newaxis,:])
    coord_y = np.multiply(np.arange(h)[:, np.newaxis],  
            np.ones((1,w), dtype=np.int))


    heat_map = np.zeros((14, h, w))
    kernel = gkern(R, 3)
    kp = []

    for i in range(14):

        reg_x = (reg[0, i*2] * R + coord_x).astype(np.int)
        reg_y = (reg[0, i*2+1] * R + coord_y).astype(np.int)

        reg_x[reg_x < 0] = 0
        reg_x[reg_x >= w] = w - 1

        reg_y[reg_y < 0] = 0
        reg_y[reg_y >= h] = h - 1

        np.add.at(heat_map[i], (reg_y, reg_x), score[i, 1])

        heat_map[i] = scipy.signal.fftconvolve(
                heat_map[i]/(R*R*3.14), kernel, 'same')

        p = heat_map[i].max()
        y, x = np.unravel_index(np.argmax(heat_map[i]), heat_map[i].shape)
        x /= im_scales[0][0]
        y /= im_scales[0][1]

        kp += [x, y, p]


    _t['im_postproc'].toc()
    #showImage(im, kp, idx)

    return kp



def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)

    output_dir = get_output_dir(imdb, net)
    kps = []

    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), \
            'im_postproc': Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):

        im = cv2.imread(imdb.image_path_at(i))
        kp = im_detect(net, im, _t, i)
        kps.append(kp)

        print 'im_detect: {:d}/{:d}  net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
              .format(i + 1, num_images, _t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                      _t['misc'].average_time)


    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(kps, f, cPickle.HIGHEST_PROTOCOL)

