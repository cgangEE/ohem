# --------------------------------------------------------
# Fast R-CNN with OHEM
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Abhinav Shrivastava
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.
"""

import sys
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from roi_data_layer.minibatch import get_minibatch, get_allrois_minibatch, get_ohem_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
from rpn.proposal_target_layer import _get_bbox_regression_labels, _compute_targets
from utils.cython_bbox import bbox_overlaps


class RepoolTargetLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

        ohem_size = cfg.TRAIN.BATCH_OHEM_SIZE

        # bbox_targets
        top[0].reshape(ohem_size , 8)
        # labels
        top[1].reshape(ohem_size, 1)
        # bbox_inside_weights
        top[2].reshape(ohem_size , 8)
        # bbox_outside_weights
        top[3].reshape(ohem_size , 8)
        assert len(top) == 4

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        rois_repool = bottom[0].data.copy()
        gt = bottom[1].data.copy()

        overlaps = bbox_overlaps(
            np.ascontiguousarray(rois_repool[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)

        fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

        labels = np.zeros(len(rois_repool), np.float32)
        labels[fg_inds] = 1.0

        bbox_target_data = _compute_targets(
            rois_repool[:, 1:5], gt[gt_assignment, :4], labels)
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, self._num_classes)

        top[0].reshape(*(bbox_targets.shape))
        top[0].data[...] = bbox_targets.astype(np.float32, copy=False)

        # classification labels
        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

        # bbox_inside_weights
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)



    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


