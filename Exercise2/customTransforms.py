#!/usr/bin/env python3

import numpy as np
import torch

class SaltNPepper(object):
    """Insert some salt and pepper grain

    percentage: Fraction of how much salt/pepper to add
    integer: both salt and pepper equally added
    tuple: (salt, pepper)-amount
    """
    def __init__(self, fraction):
        if isinstance(fraction, tuple):
            self.amount_salt = fraction[0]
            self.amount_pepper = fraction[1]
        else:
            self.amount_salt = fraction
            self.amount_pepper = fraction


    def __call__(self, img):
        shape = img.shape

        # TODO: make this work

        salt_mask = torch.trunc(self.amount_salt*torch.ones(shape,
                                                            dtype=torch.dtype.uint8) + torch.rand(shape))
        # pepper_mask = torch.trunc(self.amount_salt*torch.ones(shape) + torch.rand(shape))

        # apply salt
        img.masked_scatter_(torch.ByteTensor(salt_mask), salt_mask)

        # apply pepper
        # img.masked_scatter_(torch.ByteTensor(pepper_mask), torch.ones(shape) - pepper_mask)

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
