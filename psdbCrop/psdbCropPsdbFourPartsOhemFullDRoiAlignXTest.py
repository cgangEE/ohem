#!/usr/bin/python

import os
import numpy as np
import cPickle
import tools._init_paths
from datasets.factory import get_imdb
import cv2


def psdbCropTest():

    cache_file = 'output/pvanet_full1_DRoiAlignX_FourParts_Ohem/psdbFourParts_test/zf_faster_rcnn_iter_100000_inference/detections.pkl'
    with open(cache_file, 'rb') as fid:      
        detAll = np.array(cPickle.load(fid))

    imdb = get_imdb('psdbFourParts_2015_test')

    fVal = open('val2.txt', 'w')
    fValDetail = open('val2_detail.txt', 'w')

    for i in xrange(detAll.shape[1]):
        imagename = imdb.image_path_at(i).strip()
        im = cv2.imread(imagename)
        det = detAll[1, i]  


        for j, bbox in enumerate(det):          # fppi = 0.1: 0.94326514 
            if bbox[-1] >= 0.0:                 # fppi = 1: 0.31600133
                imageCropName = os.path.join( 'val2_' + 
                        imagename.split('/')[-1][:-4] + 
                        '_' + str(j) + '.jpg')

                fVal.write('{}\n'.format(imageCropName))
                fValDetail.write('{} {} {} {} {} {}\n'.format(
                        imageCropName, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))

                cv2.imwrite(os.path.join('image', imageCropName), 
                        im[ bbox[1] : bbox[3], bbox[0]: bbox[2] ])
                

    fVal.close()
    fValDetail.close()

if __name__ == '__main__':
    psdbCropTest()
