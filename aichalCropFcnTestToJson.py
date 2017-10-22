#!/usr/bin/env python



import tools._init_paths
from datasets.factory import get_imdb
import cPickle
import matplotlib.pyplot as plt
import numpy as np
import json, cv2



def showImage(im, boxes, keypoints):
    classToColor = ['', 'red', 'yellow', 'blue', 'magenta']
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    fig = ax.imshow(im, aspect='equal')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    thresh = 0.7

    line = [[13, 14], [14, 4], [4, 5], [5, 6], [14, 1], [1, 2], [2, 3], \
            [14, 10], [10, 11], [11, 12], [14, 7], [7, 8], [8, 9]]

    c = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    for i in xrange(boxes.shape[0]):

            bbox = boxes[i]
            if bbox[-1] < thresh:
                continue
            ax.add_patch(
                    plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor= c[i], linewidth=2.0)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:d}, {:d}'.format(int(bbox[2] - bbox[0]), 
                    int(bbox[3] - bbox[1])),
                    bbox=dict(facecolor='blue', alpha=0.2),
                    fontsize=8, color='white')

            keypoint = keypoints[i]
            for j in range(14):
                if keypoint[j * 3 + 2] == 3:
                    continue

                x, y, z = keypoint[j * 3 : (j + 1) * 3]


                ax.add_patch(
                        plt.Circle((x, y), 3,
                              fill=True,
                              color = c[i], 
                              linewidth=2.0)
                    )
            for l in line:
                i0 = l[0] - 1
                p0 = keypoint[i0 * 3 : (i0 + 1) * 3] 

                i1 = l[1] - 1
                p1 = keypoint[i1 * 3 : (i1 + 1) * 3]


                if p0[2] == 3 or p1[2] == 3:
                    continue
                
                ax.add_patch(
                        plt.Arrow(p0[0], p0[1], 
                        float(p1[0]) - p0[0], float(p1[1]) - p0[1], 
                        color = c[i])
                        )


def main(thresh, box_thresh):
    pkl = 'output/fcn/aichalCrop_test/zf_faster_rcnn_iter_100000_inference/detections.pkl'
    with open(pkl, 'rb') as fid:
        kps = cPickle.load(fid)


    pkl = 'data/aichal/aichalClusterBoxTest.pkl'
    with open(pkl, 'rb') as fid:
        boxes = cPickle.load(fid)[1]


    imdb = get_imdb('aichal_2017_test')

    idx = 0
    data = []

    for i, im_boxes in enumerate(boxes):
        image = dict()

        imname = imdb.image_path_at(i)

        imname = imname.split('/')[-1][:-4]
        image['image_id'] = imname
        kp_ann = dict()


        for j, box in enumerate(im_boxes):
            kp = kps[idx]
            p = sum(kp[2::3]) / 14
            if p >= thresh and box[-1] >= box_thresh:
                for k in range(14):
                    kp[k*3] += box[0]
                    kp[k*3+1] += box[1]
                    kp[k*3+2] = 1
                kp_ann['human' + str(j+1)] = map(int, kp)

            idx += 1

        image['keypoint_annotations'] = kp_ann
        data.append(image)

    with open('pred_fcn_test_' 
            + str(thresh) + '_' + str(box_thresh) + '.json', 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    for thresh in [0.2]:
        for box_thresh in [0.3]:
            main(thresh, box_thresh)

