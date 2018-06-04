#!/usr/bin/env python3

import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter

class RandomApplyFrlomList(object):
    """Apply randomly from a list of transformations with a given probability
        Args:
            transforms (list): list of transformations
            mandatory (list): list of mandatory transformations
            p (float): probability
    """
    def __init__(self, transforms,
                 mandatory=None,
                 p=0.5):
        assert isinstance(transforms, list)
        self.mandatory = mandatory
        self.transforms = transforms
        self.p = p

    def __call__(self, img,
                 verbose=False):
        for t in self.mandatory:
            img = t(img)

        status_str = "Applied transforms:\n"
        for t in self.transforms:
            if self.p < random.random():
                img = t(img)
                status_str += "\t" + str(t) + "\n"
        if verbose:
            print(status_str)

        return img


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
        # check whether is tuple or integer
        if isinstance(filter, tuple):
            self.w_drop = filter[0]
            self.h_drop = filter[1]
        else:
            self.w_drop = filter
            self.h_drop = filter


    def __call__(self, img):
        for n_regions in range(self.num_regions):
            channels = img.shape[0]
            w_max = img.shape[1]
            h_max = img.shape[2]

            # top left corner
            x_11 = random.randrange(0, w_max-self.w_drop)
            y_11 = random.randrange(0, h_max-self.h_drop)

            # top right corner
            x_12 = x_11 + self.w_drop
            y_12 = y_11

            # bottom right corner
            x_21 = x_11
            y_21 = y_11 + self.h_drop

            # bottom right corner
            x_22 = x_11 + self.w_drop
            y_22 = y_11 + self.h_drop

            # create masks in x and y direction
            mask_x = np.linspace(x_11,x_12,num=x_12-x_11,endpoint=False,dtype=int)
            mask_y = np.linspace(y_11,y_21,num=y_21-y_11,endpoint=False,dtype=int)

            # apply mask on every channel
            for c in range(channels):
                # apply mask in x direction
                for x in mask_x:
                    # apply mask in y direction
                    for y in mask_y:
                        # multiply with 0 to make pixel black
                        img[c][x][y] *= 0

        return img


class GaussianBlur(object):
    """Convolutes image with gaussian kernel"""
    def __init__(self, sigma = 0.5):
        self.sigma = sigma
    def __call__(self, img):
        # apply gaussian filter and convert back to tensor
        img = torch.from_numpy(gaussian_filter(img, self.sigma, mode='constant'))
        return img


class ContrastNBrightness(object):
    """Change contrast and brightness of an image"""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        return img * self.alpha + torch.ones(img.shape)*self.beta
