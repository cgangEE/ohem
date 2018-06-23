#!/usr/bin/env python
import argparse
import mxnet as mx
import os
import sys
from detect.detector_camera import Detector
from symbol.symbol_factory import get_symbol
import numpy as np

import cPickle

import tools._init_paths
from datasets.factory import get_imdb
import cv2

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        net = get_symbol(net, data_shape, num_classes=num_class, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection network demo')
    parser.add_argument('--network', dest='network', type=str, default='resnet50',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='demo image directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'),
                        type=str)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=512,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.5,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')

    parser.add_argument('--imdb', dest='imdb_name', 
                        help='image extension, optional',
                        default='psdb_2015_test',
                        type=str)

    args = parser.parse_args()
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

if __name__ == '__main__':
    args = parse_args()

    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(args.data_shape)
    else:
        prefix = args.prefix

    with open('all_boxes.pkl', 'rb') as fid:
        all_boxes = cPickle.load(fid)

    for i in range(4, 0, -1):
        all_boxes[i] = all_boxes[i-1]

    imdb = get_imdb(args.imdb_name)
    num_images = len(imdb.image_index)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes_new = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    print(imdb.num_classes)


    for i in xrange(num_images):
        print(imdb.image_path_at(i))

        im = cv2.imread(imdb.image_path_at(i))
        height, width = im.shape[:2]

        '''
        dets[:, 2] *= width
        dets[:, 3] *= height
        dets[:, 4] *= width
        dets[:, 5] *= height
        '''
        
        for j in xrange(1, imdb.num_classes): 
            boxes = all_boxes[j][i]
            if len(boxes) == 0:
                boxes = np.ones((0,6))

            boxes[:, 2] *= width
            boxes[:, 3] *= height
            boxes[:, 4] *= width
            boxes[:, 5] *= height
            boxes = boxes[:, [2,3,4,5,1]]
            all_boxes_new[j][i] = boxes


    det_file = prefix + 'detections.pkl'
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes_new, f, cPickle.HIGHEST_PROTOCOL)

