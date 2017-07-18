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

class RepoolingLayer(caffe.Layer):
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

        boxes = bottom[0].data[:, 1:5]
        box_deltas = bottom[1].data
        im_info = bottom[2].data
        im_shape = (im_info[0, 0], im_info[0, 1])

        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape)

        rois_repool = pred_boxes[:,4:]
        zeros = np.zeros((rois_repool.shape[0], 1), dtype = np.float32)
        rois_repool = np.hstack((zeros, rois_repool))


        
        top[0].reshape(*(rois_repool.shape))
        top[0].data[...] = rois_repool.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class SplitLayer(caffe.Layer):
    def setup(self, bottom, top):
        idx = 0
        ohem_size = cfg.TRAIN.BATCH_OHEM_SIZE

        top[idx].reshape(ohem_size, 4096)
        idx += 1

        top[idx].reshape(ohem_size, 4096)
        idx += 1

        top[idx].reshape(ohem_size, 4096)
        idx += 1

        top[idx].reshape(ohem_size, 4096)
        idx += 1

        assert len(top) == 4

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        fc7 = bottom[0].data
        size = fc7.shape[0] / 4

        for i in range(4):
            fc7_bbox = fc7[i * size : (i + 1) * size, : ]
            top[i].reshape(*(fc7_bbox.shape))
            top[i].data[...] = fc7_bbox.astype(np.float32, copy=False)


    def backward(self, top, propagate_down, bottom):
        diff_bbox = top[0].diff
        diff_head = top[1].diff
        diff_head_shoulder = top[2].diff
        diff_upper_body = top[3].diff
        diff = np.vstack((diff_bbox, diff_head, 
                diff_head_shoulder, diff_upper_body))

        bottom[0].diff[...] = diff.astype(np.float32, copy=False)



    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


