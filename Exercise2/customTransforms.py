#!/usr/bin/env python3

import numpy as np
import random
import torch

class SaltNPepper(object):
    """Insert some salt and pepper grain

    percentage: Fraction of how much salt/pepper to add
    integer: both salt and pepper equally added
    tuple: (salt, pepper)-amount
    """
    def __init__(self, fraction, colored=False):
        if isinstance(fraction, tuple):
            self.amount_salt = fraction[0]
            self.amount_pepper = fraction[1]
        else:
            self.amount_salt = fraction
            self.amount_pepper = fraction

        self.colored = colored


    def __call__(self, img):
        shape = img.shape

        # if the noise should be black and white: copy one mask to all colourchannels
        if self.colored:
            salt_mask = torch.trunc(self.amount_salt*torch.ones(shape) + torch.rand(shape))
            pepper_mask = torch.trunc(self.amount_salt*torch.ones(shape) + torch.rand(shape))
        else:
            shape = (1, shape[1], shape[2])
            salt_mask = torch.trunc(self.amount_salt*torch.ones(shape) + torch.rand(shape))
            pepper_mask = torch.trunc(self.amount_salt*torch.ones(shape) + torch.rand(shape))
            # we need to expand the maps to every channel
            salt_mask = torch.cat((salt_mask,salt_mask,salt_mask), dim=0)
            pepper_mask = torch.cat((pepper_mask,pepper_mask,pepper_mask), dim=0)

        print("Shape mask: {}, Shape img: {}".format(salt_mask.shape, img.shape))

        # apply salt
        img.masked_scatter_(salt_mask.byte(), salt_mask)

        # apply pepper
        img.masked_scatter_(pepper_mask.byte(), torch.ones(shape) - pepper_mask)

        return img

class GaussianNoise(object):
    """Add Gaussian Noise to a tensor"""
    def __init__(self, std, mean=0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        shape = img.shape
        mean = torch.ones(shape) * self.mean
        std = torch.ones(shape) * self.std

        img = img + torch.normal(mean, std)

        return img




class RegionDropout(object):
    """Random Region Droput with roughly 1% of image size
    integer: filter size squared, e.g. 3 -> 3x3
    tuple: (w,h) specific filter size, e.g. 3,4 -> 3x4
    """
    def __init__(self, filter=10, num_regions=1):

        self.num_regions = num_regions

        if isinstance(filter, tuple):
            self.w_drop = filter[0]
            self.h_drop = filter[1]
        else:
            self.w_drop = filter
            self.h_drop = filter


    def __call__(self, img):

        for n_regions in range(self.num_regions):
            channels = img.shape[0]
            h_max = img.shape[2]
            w_max = img.shape[1]
            x_11 = random.randrange(0, w_max-self.w_drop)
            y_11 = random.randrange(0, h_max-self.h_drop)

            x_12 = x_11 + self.w_drop
            y_12 = y_11

            x_21 = x_11
            y_21 = y_11 + self.h_drop

            x_22 = x_11 + self.w_drop
            y_22 = y_11 + self.h_drop

            # print("Left top", x_11,y_11)
            # print("Right top", x_12,y_12)
            # print("Left bot", x_21,y_21)
            # print("Right bot", x_22,y_22)

            mask_x = np.linspace(x_11,x_12,num=x_12-x_11,endpoint=False,dtype=int)
            mask_y = np.linspace(y_11,y_21,num=y_21-y_11,endpoint=False,dtype=int)
            # print("x mask",mask_x.shape)
            # print("y mask",mask_y)

            for c in range(channels):
                for x in mask_x:
                    for y in mask_y:
                        img[c][x][y] *= 0

        return img
