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

class RepoolTargetLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

        ohem_size = cfg.TRAIN.BATCH_OHEM_SIZE
        top[idx].reshape(ohem_size , 8)
        idx += 1
        assert len(top) == 1

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        rois = bottom[0].data.copy()[:, 1:5]
        gt_targets = bottom[1].data.copy()
        rois_repool = bottom[2].data.copy()
        labels = bottom[3].data.copy()

        j = 1
        gt_targets[:, j*4:(j+1)*4] *= np.array(
                cfg.TRAIN.BBOX_NORMALIZE_STDS)
        gt_targets[:, j*4:(j+1)*4] += np.array(
                cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        gt_boxes = bbox_transform_inv(rois, gt_targets)

        bbox_target_data = _compute_targets(
            rois_repool[:, 1:5], gt_boxes[:, 4:], labels)
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels(bbox_target_data, self._num_classes)

        top[0].reshape(*(bbox_targets.shape))
        top[0].data[...] = bbox_targets.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


