
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
STEER_IDX = 0 # float
GAS_IDX = 1 # float
BRAKE_IDX = 2 # float
HAND_BRAKE_IDX = 3 # boolean
REVERSE_GEAR_IDX = 4 # boolean
STEER_NOISE_IDX = 5 # float
GAS_NOISE_IDX = 6 # float
BRAKE_NOISE_IDX = 7 # float
POSITION_X_IDX = 8 # float
POSITION_Y_IDX = 9 # float
SPEED_IDX = 10 # float
COLLISION_OTHER_IDX = 11 # float
COLLISION_PEDESTRIAN_IDX = 12 # float
COLLISION_CAR_IDX = 13 # float
OPPOSITE_LANE_INTER_IDX = 14 # float
SIDEWALK_INTERSECT_IDX = 15 # float
ACCELERATION_X_IDX = 16 #float
ACCELERATION_Y_IDX = 17 # float
ACCELERATION_Z_IDX = 18 # float
PLATFORM_TIME_IDX = 19 # float
GAME_TIME_IDX = 20 # float
ORIENTATION_X_IDX = 21 # float
ORIENTATION_Y_IDX = 22 # float
ORIENTATION_Z_IDX = 23 # float
HIGH_LEVEL_COMMAND_IDX = 24 # int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
NOISE_IDX = 25 #, Boolean # ( If the noise, perturbation, is activated, (Not Used) )
CAMERA_IDX = 26 # (Which camera was used)
ANGLE_IDX = 27 # (The yaw angle for this camera)

### CONSTANTS ###

STEERING_ANGLE_IDX = 0
COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}


class H5Dataset(Dataset):
    '''
    from on data_06398.h5 ALL h5 files have 4 keys instead of 2!
    [depth, labels, rgb, targets]
    I have no idea what depth and labels are supposed to mean, since they are
    not documented. I will omit them
    '''
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.file_names = sorted(os.listdir(self.root_dir))
        self.file_names = self._check_corruption(self.file_names)

        # print(self.file_names)
        # print(len(self.file_names))
        self.file_idx = 0

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

        # print("Idx: {}, Type: {}".format(idx, type(idx)))  # TODO: Remove me
        f = h5py.File(self.root_dir + '/' + self.file_names[file_idx], 'r')

        # for magic idx numers inspect class description
        data = f['rgb']
        targets = f['targets']
        sample = {'filename' : self.file_names[file_idx],
                  'data' : data[idx],
                  'targets' : targets[idx]}

        if self.transform:
            sample = (self.transform(data[idx]),
                      torch.Tensor(targets[idx]))
        else:
            sample = (data[idx],
                      targets[idx])

        return sample


""" Just show pretty, enhanced samples"""
if  __name__=="__main__":
    # dummy composition for debugging
    composed = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,)),
                                   ContrastNBrightness(1.5,0.5),
                                   GaussianBlur(1.5),
                                   SaltNPepper(0.1),
                                   GaussianNoise(0, 0.1),
                                   RegionDropout((10, 10),10)
    ])
    un_composed = transforms.Compose([transforms.ToTensor()])

    train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    orig_train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    browser = ImageBrowser(train_set, orig_train_set)
    browser.show()
