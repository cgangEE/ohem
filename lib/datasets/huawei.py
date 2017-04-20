#!/usr/bin/python
import os
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import uuid

class huawei(imdb):
    def __init__(self, image_set, year):
        imdb.__init__(self, 'huawei_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'huawei')

        self._classes = ('background', 'pedestrian', 'car')

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        self._image_ext = '.jpg'
        self._roidb_handler = self.gt_roidb

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp9'


        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])


    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index


    def _get_annotation(self, index):
        filename  = os.path.join(self._data_path, 
                'annotation', index[6:-3]+'txt')
        f = open(filename, 'r')
        classSet = set(['person', 'car'])

        b = []
        c = []

        for line in f:
            colonPos= str.find(line,':')
            classname = line[:colonPos]
            if classname in classSet:
                x = map(float, line[colonPos+1:].strip().split())
                if len(x)>0 and max(x)>0:
                    x = np.array(x)
                    x[x<0] = 0

                    if x[0]>x[2]:
                        x[0],x[2] = x[2],x[0]
                    if x[1]>x[3]:
                        x[1],x[3] = x[3],x[1]

                    b.append(x)
                    c.append(classname=='person' and 1 or 2)



        num_objs = len(b)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ix = 0

        # Load object bounding boxes into a data frame.
        for i, box in enumerate(b):
            x1, y1, x2, y2 = box
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = c[i]
            overlaps[ix, c[i]] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def gt_roidb(self):
        '''
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        '''
        

        gt_roidb = [self._get_annotation(index)
                    for index in self.image_index]

        return gt_roidb

        '''
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)
        '''




    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb


    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    my_cox_reid = huawei('train', '2016');
    my_cox_reid.gt_roidb()
