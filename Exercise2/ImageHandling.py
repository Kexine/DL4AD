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

    # def draw_arrow(self, verbose_cmd):
    #     print(verbose_cmd)
    #     if verbose_cmd=="Straight":
    #         return plt.arrow(100,44,0,-10, width = 1, color='r')
    #     if verbose_cmd=="Follow Lane":
    #         return plt.arrow(100,44,0,-10, width = 1, color='r')
    #     if verbose_cmd=="Right":
    #         return plt.arrow(100,44,20,0, width = 1, color='r')
    #     if verbose_cmd=="Left":
    #         return plt.arrow(100,44,-20,0, width = 1, color='r')


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
        self.create_sidebyside()
        plt.imshow(self.create_sidebyside())
        plt.title("File: {}| Image {}".format(self.dataset1[self.idx]['filename'], self.idx%200 ))


        # COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}
        # STEER_IDX = 0
        # HIGH_LEVEL_COMMAND_IDX = 24
        # verbose_cmd = self.dataset1[self.idx]['targets'][HIGH_LEVEL_COMMAND_IDX]
        # self.draw_arrow(verbose_cmd)
        plt.draw()

    def create_sidebyside(self):
        img1 = self.dataset1[self.idx]['data']
        img2 = self.dataset2[self.idx]['data']

        return torch.cat((img1, img2),
                         dim = 2).numpy().transpose((1,2,0))

    def show(self):
        # plt.figure()
        plt.imshow(self.create_sidebyside())
        plt.connect('key_press_event', self.process_key)

        plt.show()
