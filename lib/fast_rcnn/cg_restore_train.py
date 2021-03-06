# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import google.protobuf.text_format
import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os
import shutil

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 solverstate=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and
            cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
            # RPN can only use precomputed normalization because there are no
            # fixed statistics to compute a priori
            assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

        if cfg.TRAIN.BBOX_REG:
            print 'Computing bounding-box regression targets...'
            self.bbox_means, self.bbox_stds = \
                    rdl_roidb.add_bbox_regression_targets(roidb)
            print 'done'

        self.solver = caffe.SGDSolver(solver_prototxt)
        if solverstate is not None:
            print ('Loading solverstate '
                   'weights from {:s}').format(solverstate)

            solverstate = solverstate.strip()
            caffemodelFull = solverstate.split('.')[0] + '.caffemodel'
            caffemodel = caffemodelFull.split('/')[-1]

            os.symlink(caffemodelFull, caffemodel)
            self.solver.restore(solverstate)
            os.remove(caffemodel)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def vis_detections(self, im, dets, pred_kp, labels = None):

        """Visual debugging of detections."""
        import matplotlib.pyplot as plt
        im = im[:, :, (2, 1, 0)]

        print('dets.shape', dets.shape)

        for i in xrange(len(dets)):
            if labels is None or labels[i] == 1.:

                fig, ax = plt.subplots(figsize=(12, 12))
                fig = ax.imshow(im, aspect='equal')
                plt.axis('off')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                print('dets.shape', dets.shape)


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




    def gao(self):
        from fast_rcnn.bbox_transform_kp import clip_boxes, bbox_transform_inv, kp_transform_inv, clip_kps

        net = self.solver.net

        im = net.blobs['data'].data.copy()
        im = im[0, :, :, :]
        im = im.transpose(1, 2, 0)
        im += cfg.PIXEL_MEANS
        im = im.astype(np.uint8, copy=False)

        rois = net.blobs['rois'].data.copy()
        boxes = rois[:, 1:5]

#        bbox_targets = net.blobs['head_targets_hard_repool'].data.copy()
        labels = net.blobs['labels'].data.copy()
        
        bbox_gt = net.blobs['bbox_targets'].data.copy()
        bbox_targets = net.blobs['bbox_pred'].data.copy()

        bbox_targets[:, 4:] *= np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_targets[:, 4:] += np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

        pred_boxes = bbox_transform_inv(boxes, bbox_targets)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
        cls_boxes = pred_boxes[:, 4:]

        kp_gt = net.blobs['kp_targets'].data.copy()
        kp_targets = net.blobs['kp_pred'].data.copy()

        kp_targets[:, :] *= np.array(cfg.TRAIN.KP_NORMALIZE_STDS)
        kp_targets[:, :] += np.array(cfg.TRAIN.KP_NORMALIZE_MEANS)

        pred_kp = kp_transform_inv(boxes, kp_targets)
        pred_kp = clip_kps(pred_kp, im.shape)

        print(boxes.shape)
        print(kp_targets.shape)
        print(pred_kp.shape)
        print(cls_boxes.shape)

        print(labels[0])
        print(bbox_targets[0])
        print(bbox_gt[0])

        print(kp_targets[0])
        print(kp_gt[0])

        print(net.blobs['kp_inside_weights'].data.copy()[0])


#        pred_kp = clip_boxes(pred_boxes, im.shape)

        self.vis_detections(im, cls_boxes, pred_kp, labels)
        exit(0)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')

        self.solver.snapshot()

        caffemodel = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        caffemodelFull = os.path.join(self.output_dir, caffemodel)

        shutil.copyfile(caffemodel, caffemodelFull)
        os.remove(caffemodel)

        solverstate = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.solverstate')
        solverstateFull = os.path.join(self.output_dir, solverstate)

        shutil.copyfile(solverstate, solverstateFull)
        os.remove(solverstate)

        return caffemodelFull


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []


        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

#            self.gao()

            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())
                

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb

def train_net(solver_prototxt, roidb, output_dir,
              solverstate=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    roidb = filter_roidb(roidb)
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       solverstate=solverstate)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
