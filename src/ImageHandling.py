#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import random
import torch

# TODO: This is only a workaround
target_idx = {'steer': 0,
              'speed': 10,
              'command': 24}


'''
speed in target_idx_raiscar means actually gas, but speed is used to change not
to much in the code
'''
target_idx_raiscar = {'command' : 0,
                      'steer' : 1,
                      'speed' : 2}


def _get_current_width():
    return (plt.gcf().get_size_inches()*plt.gcf().dpi)[0]

class ImageBrowser:

    """ Get the datasets and at which index to start."""
    def __init__(self,
                 datasets,
                 idx = None,
                 raiscar = False):

        for i in range(len(datasets) - 1):
            assert len(datasets[i]) == len(datasets[i+1]), \
"Datasets must have same length for the ImageBrowser!"

        # create a random index, if none given
        if idx is None:
            idx = random.randrange(0, len(datasets[0]))

        self.datasets = datasets

        self.__update_index(idx)

        self.side_by_side = None

        self.COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}

        self.SPEED_LIMIT_VISUAL = 5

        self.raiscar = raiscar

        # radius for circle drawing
        self.radius = 25
        self.max_speed = 30

        if self.raiscar == False:
            self.target_idx = target_idx
        else:
            self.target_idx = target_idx_raiscar

    def __update_index(self, idx):
        # truncate at beginning and end of the dataset
        idx = max(idx, 0)
        idx = min(idx, len(self.datasets[0]))

        self.idx = idx
        self.current_target = self.datasets[0][self.idx][1]

    def cmd2verbose(self):
        cmd = int(self.current_target[self.target_idx['command']])
        return self.COMMAND_DICT[cmd]

    # draw the current steering angle as arrow
    def draw_angle(self):
        rad = self.current_target[self.target_idx['steer']]
        speed = self.current_target[self.target_idx['speed']]
        # speed_norm = self.radius /(speed+self.radius) * self.radius
        speed_norm = speed/(self.radius)
        if speed>self.SPEED_LIMIT_VISUAL:
            dx = np.cos(rad - np.pi/2) * speed
            dy = np.sin(rad - np.pi/2) * speed
            return plt.arrow(get_current_width(),
                             88,dx,dy, width = 1, color='g')
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
        file_idx = int(self.idx/200)
        image_idx =  self.idx%200

        filename = "data_{:05d}.h5".format(file_idx)

        target = self.current_target

        st_angle = target[self.target_idx['steer']]
        speed = target[self.target_idx['speed']]
        verbose_cmd = self.cmd2verbose()
        plt.title("File: {}| Image {}\n Steering Angle: {:.4f}| Speed: {:.2f}\n Command {}"\
                  .format(filename, image_idx, st_angle,speed, verbose_cmd))


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
            self.__update_index(self.idx - 1)
        elif event.key == 'right' or event.key == 'up':
            self.__update_index(self.idx + 1)
        elif event.key == 'r':
            self.__update_idx(random.randrange(0, len(self.dataset1)))
            print("New Index: {}".format(self.idx))
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
        images = [data[self.idx][0] for data in self.datasets]

        return torch.cat(images,
                         dim = 2).numpy().transpose((1,2,0))

    def show(self):
        self.make_title()
        plt.imshow(self.create_sidebyside(), interpolation='nearest')
        self.draw_arrow()
        self.draw_circle()
        plt.connect('key_press_event', self.process_key)

        plt.show()

        return
