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

        self.HIGH_LEVEL_COMMAND_IDX = 24
        self.STEER_IDX = 0

    def make_title(self):
        filename = self.dataset1[self.idx]['filename']
        image_idx =  self.idx%200
        st_angle = self.dataset1[self.idx]['targets'][self.STEER_IDX]
        return plt.title("File: {}| Image {}| Steering Angle: {:.4f}".format(
                    filename, image_idx, st_angle))


    def draw_arrow(self, cmd):
        COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}
        verbose_cmd = COMMAND_DICT[cmd]
        print(verbose_cmd)
        if verbose_cmd=="Straight":
            return plt.arrow(300,44,0,-10, width = 1, color='r')
        if verbose_cmd=="Follow Lane":
            return #plt.arrow(300,44,0,-10, width = 1, color='r')
        if verbose_cmd=="Right":
            return plt.arrow(300,44,20,0, width = 1, color='r')
        if verbose_cmd=="Left":
            return plt.arrow(300,44,-20,0, width = 1, color='r')


    def process_key(self, event):
        if event.key == 'left' or event.key == 'down':
            self.idx = max(self.idx - 1, 0)
        elif event.key == 'right' or event.key == 'up':
            self.idx = min(self.idx + 1, len(self.dataset1))
        elif event.key == 'r':
            self.idx = random.randrange(0, len(self.dataset1))
            print("New Index: {}".format(self.idx))
        elif event.key == 'f10' or event.key == 'q':
            plt.close()
        else:
            return

        # replot
        plt.clf()
        self.create_sidebyside()
        plt.imshow(self.create_sidebyside())
        self.make_title()
        self.draw_arrow(self.dataset1[self.idx]['targets'][self.HIGH_LEVEL_COMMAND_IDX])
        plt.draw()

    def create_sidebyside(self):
        img1 = self.dataset1[self.idx]['data']
        img2 = self.dataset2[self.idx]['data']

        return torch.cat((img1, img2),
                         dim = 2).numpy().transpose((1,2,0))

    def show(self):
        self.make_title()
        plt.imshow(self.create_sidebyside())
        self.draw_arrow(self.dataset1[self.idx]['targets'][self.HIGH_LEVEL_COMMAND_IDX])
        plt.connect('key_press_event', self.process_key)

        plt.show()
