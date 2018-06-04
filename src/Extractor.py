#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 1
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

#!/usr/bin/env python3
from __future__ import print_function, division
import h5py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os, os.path
import random
import matplotlib.pyplot as plt

from customTransforms import *

# from FallbackGUI import PrimitiveGUI
from ImageHandling import ImageBrowser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# from ImageHandling import ImageBrowser

import warnings
torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ignore warnings
warnings.filterwarnings("ignore")

IMAGES_PER_FILE = 200

# Target Array indices
target_idx = {'steer': 0,  # float
              'gas': 1,  # float
              'brake': 2,  # float
              'handbrake': 3,  # boolean
              'reverse gear': 4,  # boolean
              'steer noise': 5,  # float
              'gas noise': 6,  # float
              'brake noise': 7,  #
              'pos x': 8,  # float
              'pos y': 9,  # float
              'speed': 10,  # float
              'collision other': 11,  # float
              'collision pedestrian': 12,  # float
              'collision car': 13,  # float
              'opposite lane intersection': 14,  # float
              'sidewalk intersection': 15,  # float
              'acc x': 16,  # float
              'acc y': 17,  # float
              'acc z': 18,  # float
              'platform time': 19,  # float
              'game time': 20,  # float
              'orientation x': 21,  # float
              'orientation y': 22,  # float
              'orientation z': 23,  # float
              'command': 24,  # int
              'noise': 25,  # boolean
              'camera number': 26,  #
              'camera yaw': 27  #
}

### CONSTANTS ###

COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}


class H5Dataset(Dataset):
    '''
    from on data_06398.h5 ALL h5 files have 4 keys instead of 2!
    [depth, labels, rgb, targets]
    I have no idea what depth and labels are supposed to mean, since they are
    not documented. I will omit them
    '''
    def __init__(self, root_dir,
                 transform=None,
                 images_per_file=IMAGES_PER_FILE):
        self.root_dir = root_dir
        self.transform = transform

        self.file_names = sorted(os.listdir(self.root_dir))
        self.file_names = self._check_corruption(self.file_names)

        # print(self.file_names)
        # print(len(self.file_names))
        self.file_idx = 0
        self.current_file = None

    def _check_corruption(self,file_names):
        crpt_idx = []
        old_length = len(file_names)
        for idx, val in enumerate(file_names):
            # check if h5 is corrupted by checking for file signature exception
            try:
                f = h5py.File(self.root_dir + '/' + file_names[idx], 'r')
            except OSError:
                print("File {} is corrupted and will be removed from list".format(file_names[idx]))
                f.close()
                # if corrupte file found, save index
                crpt_idx.append(idx)
        # delete corrupted file names from the file list
        for i in crpt_idx:
            del file_names[i]
        new_length = len(file_names)
        print("{} files have been removed from the Training Set".format(old_length-new_length))
        return file_names


    def __len__(self):
        return len(self.file_names)*IMAGES_PER_FILE

    def __getitem__(self, idx):
        # The input idx seems sequential but we're actually going through the
        # images in each file and going through all the files in order.
        # Note: Danger! This >>WILL<< break, if any h5 file doesn't have
        # >>EXACTLY<< 200 images
        raw_idx = idx
        file_idx = int(idx / IMAGES_PER_FILE)
        idx = idx % IMAGES_PER_FILE

        if (file_idx != self.file_idx
            or self.current_file is None):
            self.current_file = h5py.File(self.root_dir + '/' + self.file_names[file_idx], 'r')
            self.file_idx = file_idx

        # for magic idx numbers inspect class description
        data = self.current_file['rgb'][idx]
        targets = self.current_file['targets'][idx]

        if self.transform:
            sample = (self.transform(data),
                      torch.Tensor(targets))
        else:
            sample = (data,
                      targets)

        return sample


""" Just show pretty, enhanced samples"""
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain')
    args = parser.parse_args()

    traindata_path = args.train

    # dummy composition for debugging
    composed = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   ContrastNBrightness(1.5,0.5),
                                   GaussianBlur(1.5),
                                   SaltNPepper(0.1),
                                   GaussianNoise(0, 0.1),
                                   RegionDropout((10, 10),10)])

    randomized = RandomApplyFromList([ContrastNBrightness(1.5,0.5),
                                      GaussianBlur(1.5),
                                      SaltNPepper(0.1),
                                      GaussianNoise(0, 0.1),
                                      RegionDropout((10, 10),10)],
                                     mandatory=[transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])
    un_composed = transforms.Compose([transforms.ToTensor()])

    train_set = H5Dataset(root_dir = traindata_path,
                          transform=randomized)

    orig_train_set = H5Dataset(root_dir = traindata_path,
                               transform=un_composed)

    browser = ImageBrowser(train_set, orig_train_set)
    browser.show()


if __name__ == "__main__":
    main()
