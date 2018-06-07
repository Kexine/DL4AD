#!/usr/bin/env python3

import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as ptt
from scipy.ndimage import gaussian_filter

class RandomApplyFromList(object):
    """Apply randomly from a list of transformations with a given probability
        Args:
            transforms (list): list of transformations
            mandatory (list): list of mandatory transformations
            p (float): probability
    """
    def __init__(self, transforms,
                 mandatory=None,
                 p=0.5,
                 verbose=False,
                 normalize=True,
                 std=None):
        assert isinstance(transforms, list)
        self.mandatory = mandatory
        self.transforms = transforms
        self.p = p
        self.verbose = verbose
        self.normalize = normalize
        self.std = std

    def __call__(self, img):
        if self.normalize:
            img = ptt.ToTensor()(img)
            if self.std:
                std = self.std
            else:
                std = img.std()
            img = ptt.Normalize([img.mean()], [std])(img)
        elif self.mandatory:
            for t in self.mandatory:
                img = t(img)

        status_str = "Applied transforms:\n"
        for t in self.transforms:
            if self.p > random.random():
                img = t(img)
                status_str += "\t" + str(t) + "\n"
        if self.verbose:
            print(status_str)

        return img


class JustNormalize(object):
    def __init__(self, std=None):
        self.std = std

    def __call__(self, img):
        img = ptt.ToTensor()(img)
        if self.std:
            std = self.std
        else:
            std = img.std()
        img = ptt.Normalize([img.mean()], [std])(img)

        return img


class SaltNPepper(object):
    """Insert some salt and pepper grain

    percentage: Fraction of how much salt/pepper to add
    integer: both salt and pepper equally added
    tuple: (salt, pepper)-amount
    """
    def __init__(self,
                 fraction = None,
                 colored=False):
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


    def __str__(self):
        return "Salt and Pepper transform, salt: {}, pepper:  ".format(self.amount_salt,
                                                                       self.amount_pepper)

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

    def __str__(self):
        return "Gaussian Noise transform, μ= {}, σ= {} ".format(self.mean,
                                                                self.std)


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

            # bottom right corner
            x_22 = x_11 + self.w_drop
            y_22 = y_11 + self.h_drop

            # multiply with 0 to make pixel black
            img[0:channels, x_11:x_22, y_11:y_22] = 0

        return img


    def __str__(self):
        return "Region Dropout"


class GaussianBlur(object):
    """Convolutes image with gaussian kernel"""
    def __init__(self, sigma = 0.5):
        self.sigma = sigma
    def __call__(self, img):
        # apply gaussian filter and convert back to tensor
        img = torch.from_numpy(gaussian_filter(img, self.sigma, mode='constant'))
        return img
    def __str__(self):
        return "Gaussian Blur, σ= {}".format(self.sigma)


class ContrastNBrightness(object):
    """Change contrast and brightness of an image"""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        return img * self.alpha + torch.ones(img.shape)*self.beta

    def __str__(self):
        return "Contrast n Brigntess, α= {}, β= {}".format(self.alpha, self.beta)
