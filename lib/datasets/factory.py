# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.cox_reid import cox_reid
from datasets.psdb import psdb
from datasets.psdb3 import psdb3
from datasets.psdbVeh import psdbVeh
from datasets.flickr import flickr
from datasets.tsinghua import tsinghua
from datasets.huawei import huawei
from datasets.sjtu_ia import sjtu_ia
from datasets.fish import fish

from datasets.detvipl import detvipl
from datasets.detviplV4 import detviplV4
from datasets.detviplV4d2 import detviplV4d2
from datasets.detviplV4d2X import detviplV4d2X
from datasets.detviplV4d2XX import detviplV4d2XX

import numpy as np

# Set up detviplV4d2XX_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'detviplV4d2XX_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detviplV4d2XX(split, year))


# Set up detviplV4d2X_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'detviplV4d2X_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detviplV4d2X(split, year))


# Set up detviplV4d2_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'detviplV4d2_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detviplV4d2(split, year))


# Set up detviplV4_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'detviplV4_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detviplV4(split, year))


# Set up detvipl_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'detvipl_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: detvipl(split, year))


# Set up fish_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test', 'val']:
        name = 'fish_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: fish(split, year))

# Set up sjtu_ia_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'sjtu_ia_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: sjtu_ia(split, year))

# Set up huawei_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'huawei_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: huawei(split, year))

# Set up tsinghua_<year>_<split> using selective search "fast" mode
for year in ['2014']:
    for split in ['test']:
        name = 'tsinghua_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: tsinghua(split, year))


# Set up psdb3_<year>_<split> using selective search "fast" mode
for year in ['2016']:
    for split in ['train', 'test']:
        name = 'flickr_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: flickr(split, year))

# Set up psdb3_<year>_<split> using selective search "fast" mode
for year in ['2015']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'psdb3_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: psdb3(split, year))


# Set up psdbVeh_<year>_<split> using selective search "fast" mode
for year in ['2015']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'psdbVeh_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: psdbVeh(split, year))

# Set up psdb_<year>_<split> using selective search "fast" mode
for year in ['2015']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'psdb_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: psdb(split, year))


# Set up psdb1_<year>_<split> using selective search "fast" mode
for year in ['2015']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'psdb1_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: psdb1(split, year))

# Set up cox_<year>_<split> using selective search "fast" mode
for year in ['2011']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'cox_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: cox_reid(split, year))


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
