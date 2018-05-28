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
        self.SPEED_IDX = 10 # float
        self.COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}

        self.SPEED_LIMIT_VISUAL = 5

        # radius for circle drawing
        self.radius = 25
        self.max_speed = 30

    def cmd2verbose(self):
        cmd = self.dataset1[self.idx]['targets'][self.HIGH_LEVEL_COMMAND_IDX]
        return self.COMMAND_DICT[cmd]

    # draw the current steering angle as arrow
    def draw_angle(self):
        rad = self.dataset1[self.idx]['targets'][self.STEER_IDX]
        speed = self.dataset1[self.idx]['targets'][self.SPEED_IDX]
        # speed_norm = self.radius /(speed+self.radius) * self.radius
        print(speed)
        speed_norm = speed/(self.radius)
        print(speed_norm)
        if speed>self.SPEED_LIMIT_VISUAL:
            dx = np.cos(rad - np.pi/2) * speed
            dy = np.sin(rad - np.pi/2) * speed
            return plt.arrow(300,88,dx,dy, width = 1, color='g')
        else:
            dx = np.cos(rad - np.pi/2) * self.SPEED_LIMIT_VISUAL
            dy = np.sin(rad - np.pi/2) * self.SPEED_LIMIT_VISUAL
            # magenta arrow if speed is below visual threshold, only for visualization
            return plt.arrow(300,88,dx,dy, width = 1, color='m', linestyle='dashed')


    # plot a circle to help visualizing the current steering angle
    def draw_circle(self):
        circ = plt.Circle((300, 88), self.radius, color='g', fill=False, linestyle='dashed')
        fig = plt.gcf()
        ax = fig.gca()
        # draw angle arrow in circle
        self.draw_angle()
        return ax.add_artist(circ)

    # create title for diagram with meta information
    def make_title(self):
        filename = self.dataset1[self.idx]['filename']
        image_idx =  self.idx%200
        st_angle = self.dataset1[self.idx]['targets'][self.STEER_IDX]
        speed = self.dataset1[self.idx]['targets'][self.SPEED_IDX]
        verbose_cmd = self.cmd2verbose()
        return plt.title("File: {}| Image {}\n Steering Angle: {:.4f}| Speed: {:.2f}\n Command {}".format(
                    filename, image_idx, st_angle,speed, verbose_cmd))

    def draw_arrow(self):
        verbose_cmd = self.cmd2verbose()
        if verbose_cmd=="Straight":
            return plt.arrow(300,44,0,-10, width = 1, color='r')
        if verbose_cmd=="Right":
            return plt.arrow(300,44,20,0, width = 1, color='r')
        if verbose_cmd=="Left":
            return plt.arrow(300,44,-20,0, width = 1, color='r')
        if verbose_cmd=="Follow Lane":
            # follow lane gets no special symbol since we conside this to be
            # the default state of command
            return #plt.arrow(300,44,0,-10, width = 1, color='r')

    def process_key(self, event):
        if event.key == 'left' or event.key == 'down':
            self.idx = max(self.idx - 1, 0)
        elif event.key == 'right' or event.key == 'up':
            self.idx = min(self.idx + 1, len(self.dataset1))
        elif event.key == 'r':
            self.idx = random.randrange(0, len(self.dataset1))
            print("New Index: {}".format(self.idx))
        elif event.key == '+':
            self.scaling_factor += 0.1
        elif event.key == '-':
            self.scaling_factor -= 0.1
        else:
            return

        # replot
        plt.clf()
        self.make_title()
        self.draw_arrow()
        self.draw_circle()
        plt.imshow(self.create_sidebyside(), interpolation='nearest')

        plt.draw()

    def create_sidebyside(self):
        img1 = self.dataset1[self.idx]['data']
        img2 = self.dataset2[self.idx]['data']
        result = torch.cat((img1, img2),
                         dim = 2).numpy() # .transpose((1,2,0))

        result = torch.cat((img1, img2),
                         dim = 2).numpy().transpose((1,2,0))

        return torch.cat((img1, img2),
                         dim = 2).numpy().transpose((1,2,0))

    def show(self):
        self.make_title()
        plt.imshow(self.create_sidebyside(), interpolation='nearest')
        self.draw_arrow()
        self.draw_circle()
        plt.connect('key_press_event', self.process_key)

        plt.show()

        return
