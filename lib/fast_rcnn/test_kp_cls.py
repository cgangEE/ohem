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

def im_detect(net, im, _t, boxes=None):
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
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

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
    #blobs_out = net.forward(**forward_kwargs)





    _t['im_postproc'].tic()
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    scores = blobs_out['cls_prob'].copy()
    kp_scores = blobs_out['kp_cls_prob'].copy()
    kp_scores = kp_scores[:, 1].reshape(-1, 14)


    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        box_deltas[:, 4:] *= np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        box_deltas[:, 4:] += np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)

        kp_deltas = blobs_out['kp_pred']
        kp_deltas[:, :] *= np.array(cfg.TRAIN.KP_NORMALIZE_STDS)
        kp_deltas[:, :] += np.array(cfg.TRAIN.KP_NORMALIZE_MEANS)
        pred_kp = kp_transform_inv(boxes, kp_deltas)
        pred_kp = clip_kps(pred_kp, im.shape)

    _t['im_postproc'].toc()

    return scores, pred_boxes, pred_kp, kp_scores

def vis_detections(im, dets, pred_kp, labels = None):

    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    print('dets.shape', dets.shape)

    for i in xrange(len(dets)):
        if labels is None or labels[i] == 1.:
            bbox = dets[i]
            kp = pred_kp[i]

            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
                    )
            ax.text(bbox[0], bbox[1] - 2,
                '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

            for j in range(14):
                x, y = kp[j * 2 : (j + 1) * 2]
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

    plt.show('x')


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



def showImage(im, boxes, scores, kps, kp_scores, imageId):

    print('boxes.shape', boxes.shape)
    print('scores.shape', scores.shape)
    print('kps.shape', kps.shape)
    print('kp_scores.shape', kp_scores.shape)

    ind = np.argsort(scores)[::-1]
    boxes = boxes[ind]
    scores = scores[ind]
    kps = kps[ind]
    kp_scores = kp_scores[ind]

    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]

    thresh = 0.5

    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in xrange(boxes.shape[0]):

        fig, ax = plt.subplots(figsize=(12, 12))
        fig = ax.imshow(im, aspect='equal')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        bbox = boxes[i]

        ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor= c[0], linewidth=2.0)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:d}, {:d}, {:.3f}'.format(int(bbox[2] - bbox[0]), 
                int(bbox[3] - bbox[1]), scores[i]),
                bbox=dict(facecolor='blue', alpha=0.2),
                fontsize=8, color='white')

        keypoint = kps[i]
        for j in range(14):
            x, y = keypoint[j * 2 : (j + 1) * 2]

            ax.add_patch(
                    plt.Circle((x, y), 3,
                          fill=True,
                          color = c[1], 
                          linewidth=2.0)
                )

            ax.text(x, y - 2, '{:.3f}'.format(kp_scores[i, j]),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

        for l in line:
            i0 = l[0] - 1
            p0 = keypoint[i0 * 2 : (i0 + 1) * 2] 

            i1 = l[1] - 1
            p1 = keypoint[i1 * 2 : (i1 + 1) * 2]
            
            ax.add_patch(
                    plt.Arrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], 
                    color = c[2])
                    )

        plt.savefig(str(imageId) + '_' + str(i), bbox_inches='tight', pad_inches=0)
        



def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    all_kps = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        scores, boxes, kps, kp_scores = im_detect(net, im, _t, box_proposals)


        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes):

            showImage(im, boxes[:, 4:8], scores[:, 1], kps, kp_scores, i)
            if i == 10:
                exit(0)

            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_kps = kps[inds, :]
            cls_kp_scores = kp_scores[inds, :]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)

            keep = nms(cls_dets, cfg.TEST.NMS)

            dets_NMSed = cls_dets[keep, :]
            pred_kp = cls_kps[keep, :]
            
            if cfg.TEST.BBOX_VOTE:
                cls_dets = bbox_vote(dets_NMSed, cls_dets)
            else:
                cls_dets = dets_NMSed


            if vis:
                vis_detections(im, cls_dets, pred_kp)

            all_boxes[j][i] = cls_dets
            all_kps[j][i] = pred_kp


        print 'im_detect: {:d}/{:d}  net {:.3f}s  preproc {:.3f}s  postproc {:.3f}s  misc {:.3f}s' \
              .format(i + 1, num_images, _t['im_net'].average_time,
                      _t['im_preproc'].average_time, _t['im_postproc'].average_time,
                      _t['misc'].average_time)

    results = {"boxes" : all_boxes, 
                "all_kps" : all_kps}

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(results, f, cPickle.HIGHEST_PROTOCOL)

#    print 'Evaluating detections'
#    imdb.evaluate_detections(all_boxes, output_dir)
