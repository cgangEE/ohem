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

class RepoolTargetLayer(caffe.Layer):
    def setup(self, bottom, top):

        idx = 0
        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

        ohem_size = cfg.TRAIN.BATCH_OHEM_SIZE
        top[idx].reshape(ohem_size , 5)
        idx += 1
        assert len(top) == 1

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        rois = bottom[0].data[:, 1:5]
        box_targets = bottom[1].data

        gt_boxes = bbox_transform_inv(rois, box_targets)
        gt_boxes = gt_boxes[:, 4:]


        zeros = np.zeros((gt_boxes.shape[0], 1), dtype = np.float32)
        gt_boxes = np.hstack((zeros, gt_boxes))

        print('rois.shape', rois.shape)
        print('box_targets.shape', box_targets.shape)
        print('gt_boxes.shape', gt_boxes.shape)
        exit(0)

        top[0].reshape(*(gt_boxes.shape))
        top[0].data[...] = gt_boxes.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


