#!/usr/bin/python

from datasets.psdbFourParts import *
from matplotlib import pyplot as plt

def plot(name, data):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.hist(list(data), 50, facecolor='green')
    ax.set_xlabel(name, fontsize=20)
    ax.set_ylabel('#boxes', fontsize=20)
    plt.savefig(name, bbox_inches='tight', pad_inches=0)
    

def stat():
    psdb = psdbFourParts('train', '2015')
    roidb = psdb.gt_roidb()

    partName = ['boxes', 'head', 'head_shoulder', 'upper_body']
    ratio_w = {}
    ratio_h = {}
    
    for roi in roidb:
        for i, b in enumerate(roi['boxes']):             
            [x1, y1, x2, y2] = b
            pW = x2 - x1 + 1
            pH = y2 - y1 + 1
            for j, part in enumerate(partName):
                [x1, y1, x2, y2] = roi[part][i]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if j not in ratio_w:
                    ratio_w[j] = []
                    ratio_h[j] = []

                ratio_w[j].append(float(w) / pW)
                ratio_h[j].append(float(h) / pH)

    for i, part in enumerate(partName):            
        if (i):
            plot(part + '._w_ratio.png', ratio_w[i])
            plot(part + '._h_ratio.png', ratio_h[i])

if __name__ == '__main__':
    stat()
