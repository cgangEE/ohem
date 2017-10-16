# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform_kp import bbox_transform, kps_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        batch_size = cfg.TRAIN.BATCH_SIZE

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(batch_size, 5)
        # labels
        top[1].reshape(batch_size, 1)
        # bbox_targets
        top[2].reshape(batch_size, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(batch_size, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(batch_size, self._num_classes * 4)

        #---------- _cg_ added kp ---------------------

        # kp_targets
        top[5].reshape(batch_size, 28)
        # kp_inside_weights
        top[6].reshape(batch_size, 28)
        # kp_outside_weights
        top[7].reshape(batch_size, 28)
        # kp_labels
        top[8].reshape(batch_size * 14, 1)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        gt_kps = bottom[2].data

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images

        '''
        print('rois_per_image', rois_per_image)
        print('cfg.TRAIN.BATCH_SIZE', cfg.TRAIN.BATCH_SIZE)
        '''

        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights, \
            kp_targets, kp_inside_weights, kp_labels  = _sample_rois(
                        all_rois, gt_boxes, gt_kps, fg_rois_per_image,
                        rois_per_image, self._num_classes)

        if DEBUG:
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_targets
        top[2].reshape(*bbox_targets.shape)
        top[2].data[...] = bbox_targets

        # bbox_inside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*bbox_inside_weights.shape)
        top[4].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)

        # kp_targets
        top[5].reshape(*kp_targets.shape)
        top[5].data[...] = kp_targets

        # kp_inside_weights
        top[6].reshape(*kp_inside_weights.shape)
        top[6].data[...] = kp_inside_weights

        # kp_outside_weights
        top[7].reshape(*kp_inside_weights.shape)
        top[7].data[...] = np.array(kp_inside_weights > 0).astype(np.float32)

        # kp_labels
        kp_labels = kp_labels.reshape(kp_labels.shape[0] * kp_labels.shape[1], 1)

        top[8].reshape(*kp_labels.shape)
        top[8].data[...] = kp_labels


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights




def _get_kp_regression_labels(kp_target_data, gt_kps):

    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = kp_target_data[:, 0]
    kp_targets = np.zeros((clss.size, 28), dtype=np.float32)
    kp_inside_weights = np.zeros(kp_targets.shape, dtype=np.float32)
    kp_labels = np.ones((clss.size, 14), dtype=np.float32) * -1

    inds = np.where(clss > 0)[0]

    for ind in inds:
        kp_targets[ind] = kp_target_data[ind, 1:]
        for i in range(14):

            #kp_labels[ind, i] = gt_kps[ind, i * 3 + 2]

            if gt_kps[ind, i * 3 + 2] <= 2:
                kp_labels[ind, i] = 1
            else:
                kp_labels[ind, i] = 0

            if gt_kps[ind, i * 3 + 2] <= 2:
                kp_inside_weights[ind, i*2: i*2+2] = cfg.TRAIN.KP_INSIDE_WEIGHTS

    return kp_targets, kp_inside_weights, kp_labels



def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _compute_kp_targets(ex_rois, gt_kps, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_kps.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_kps.shape[1] == 42

    targets = kps_transform(ex_rois, gt_kps)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev

        targets = ((targets - np.array(cfg.TRAIN.KP_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.KP_NORMALIZE_STDS))

    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, gt_kps, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    
    #---------- _cg_ added kp ---------------------

    kp_target_data = _compute_kp_targets(
        rois[:, 1:5], gt_kps[gt_assignment[keep_inds]], labels)
    
    kp_targets, kp_inside_weights, kp_labels = \
        _get_kp_regression_labels(kp_target_data, gt_kps[gt_assignment[keep_inds]])

    return labels, rois, bbox_targets, bbox_inside_weights, \
            kp_targets, kp_inside_weights, kp_labels
