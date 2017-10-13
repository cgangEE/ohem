#!/usr/bin/python

import tools._init_paths
import cPickle
import os
from fast_rcnn.config import cfg
import numpy as np
from datasets.factory import get_imdb
import matplotlib.pyplot as plt
import matplotlib
import cv2
import random
from aichalClusterValShowBox import showImage

matplotlib.rcParams.update({'figure.max_open_warning':0})




def showBox(image_set):
    cache_file =  \
        'output/kp/aichal_val/zf_faster_rcnn_iter_100000_inference/detections.pkl'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            results = cPickle.load(fid)
    else:
        print(cache_file + ' not found')

    imdb = get_imdb(image_set)
    num_images = len(imdb.image_index)

    boxes = np.array(results['boxes'])
    kps = results['all_kps']


    for i in xrange(num_images):
        if i % 3000 == 0:
            im_name = imdb.image_path_at(i)
            im = cv2.imread(im_name)
            
            showImage(im, boxes[1][i], kps[1][i])
            plt.savefig(str(i), bbox_inches='tight', pad_inches=0)





if __name__ == '__main__':
    showBox('aichal_2017_val')
    
