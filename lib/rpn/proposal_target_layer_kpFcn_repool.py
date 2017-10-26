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
from fast_rcnn.bbox_transform_kpFcn import bbox_transform, kps_transform
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

        # kp_targets
        top[0].reshape(batch_size * 14, 1, 192, 192)


    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)

        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data
        gt_kps = bottom[2].data

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'


        kp_targets = _sample_rois(
                        all_rois, gt_boxes, gt_kps, self._num_classes)


        # kp_targets
        top[0].reshape(*kp_targets.shape)
        top[0].data[...] = kp_targets



    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass



def _compute_kp_targets(ex_rois, gt_kps, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_kps.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_kps.shape[1] == 42

    kp_labels = kps_transform(ex_rois, gt_kps, labels)

    return kp_labels




def _sample_rois(all_rois, gt_boxes, gt_kps, num_classes):
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
    bg_inds = np.where(max_overlaps < cfg.TRAIN.KP_FCN_FG_THRESH)
    labels[bg_inds] = 0

    
    #---------- _cg_ added kp ---------------------

    kp_targets = _compute_kp_targets(
        all_rois[:, 1:5], gt_kps[gt_assignment], labels)


    return kp_targets
