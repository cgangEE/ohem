#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt

im = cv2.imread('0.png')
im = im[:, :, (2, 1, 0)]
fig, ax = plt.subplots(figsize=(12, 12))
fig = ax.imshow(im, aspect='equal')

ax.add_patch(
        plt.Rectangle((100, 100),
              300, 300,
              fill=False,
              edgecolor= 'y', linewidth=2.0)
)

ax.add_patch(
        plt.Circle((100, 100), 10,
              fill=True,
              edgecolor= 'y', linewidth=2.0)
)

plt.show()


