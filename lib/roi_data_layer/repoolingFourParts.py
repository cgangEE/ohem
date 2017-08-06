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

        for i in range(4):
            top[idx].reshape(ohem_size, 5)
            idx += 1

        top[idx].reshape(ohem_size * 4, 5)
        idx += 1
        assert len(top) == 5

    def forward(self, bottom, top):
        """Compute loss, select RoIs using OHEM. Use RoIs to get blobs and copy them into this layer's top blob vector."""

        boxes = bottom[0].data[:, 1:5]
        deltas = []

        for i in range(1, 5):
            deltas.append(bottom[i].data)
        '''
        box_deltas = bottom[1].data
        head_deltas = bottom[2].data
        head_shoulder_deltas = bottom[3].data
        upper_body_deltas = bottom[4].data
        '''

        im_info = bottom[5].data
        im_shape = (im_info[0, 0], im_info[0, 1])

        j = 1
        for i in range(4):
            deltas[i][:, j*4:(j+1)*4] *= np.array(
                    cfg.TRAIN.BBOX_NORMALIZE_STDS)
            deltas[i][:, j*4:(j+1)*4] += np.array(
                    cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            deltas[i]= bbox_transform_inv(boxes, deltas[i])
            deltas[i] = clip_boxes(deltas[i], im_shape)
            deltas[i] = deltas[i][:, 4:]


            zeros = np.zeros((deltas[i].shape[0], 1), dtype = np.float32)
            deltas[i] = np.hstack((zeros, deltas[i]))

            top[i].reshape(*(deltas[i].shape))
            top[i].data[...] = deltas[i].astype(np.float32, copy=False)


        rois_repool = np.vstack((deltas[0], deltas[1],
                deltas[2], deltas[3]))
        
        top[4].reshape(*(rois_repool.shape))
        top[4].data[...] = rois_repool.astype(np.float32, copy=False)


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
        diff = np.vstack((top[0].diff, top[1].diff, 
                top[2].diff, top[3].diff))

        bottom[0].diff[...] = diff.astype(np.float32, copy=False)



    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


