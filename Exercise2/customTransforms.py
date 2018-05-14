#!/usr/bin/env python3

import numpy as np
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
