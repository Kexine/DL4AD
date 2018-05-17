#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import torch

class ImageBrowser:

    """ Get the datasets and at which index to start."""
    def __init__(self, dataset1, dataset2, idx = None):

        assert len(dataset1) == len(dataset2), \
"Datasets must have same length for the ImageBrowser!"

        # create a random index, if none given
        if idx is None:
            idx = random.randrange(0, len(dataset1))
        self.idx = idx

        self.dataset1 = dataset1
        self.dataset2 = dataset2

        self.side_by_side = None

    def process_key(self, event):
        if event.key == 'left' or event.key == 'down':
            self.idx = max(self.idx - 1, 0)
        elif event.key == 'right' or event.key == 'up':
            self.idx = min(self.idx + 1, len(self.dataset1))
        elif event.key == 'r':
            self.idx = random.randrange(0, len(dataset1))
            print("New Index: {}".format(self.idx))
        elif event.key == 'f10' or event.key == 'q':
            plt.close()
        else:
            return

        # replot
        self.create_sidebyside()
        plt.imshow(self.create_sidebyside())

        plt.draw()

    def create_sidebyside(self):
        img1 = self.dataset1[self.idx]['data']
        img2 = self.dataset2[self.idx]['data']

        return torch.cat((img1, img2),
                         dim = 2).numpy().transpose((1,2,0))

    def show(self):
        plt.imshow(self.create_sidebyside())
        plt.connect('key_press_event', self.process_key)

        plt.show()
