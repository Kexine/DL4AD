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
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch import randperm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os, os.path
import random
import matplotlib.pyplot as plt
import collections

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

target_idx_raiscar = {'command' : 0,
                      'steer' : 1,
                      'gas' : 2}

### CONSTANTS ###

COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}
MAX_QUEUE_LENGTH = 400

class H5Dataset(Dataset):
    '''
    from on data_06398.h5 ALL h5 files have 4 keys instead of 2!
    [depth, labels, rgb, targets]
    I have no idea what depth and labels are supposed to mean, since they are
    not documented. I will omit them
    '''
    def __init__(self, root_dir,
                 transform=None,
                 images_per_file=IMAGES_PER_FILE,
                 raiscar = False):
        self.root_dir = root_dir
        self.transform = transform

        self.file_names = sorted(os.listdir(self.root_dir))
        self.file_names = self._check_corruption(self.file_names)

        # List with all the filehandles
        self.file_handle_dict = {}

        # Queue storing the order of the filehandles in RAM
        self.file_idx_queue = collections.deque()

        self.raiscar = raiscar

        if raiscar==True:
            self.target_idx = target_idx_raiscar
        else:
            self.target_idx = target_idx

    def _check_corruption(self,file_names):
        crpt_idx = []
        old_length = len(file_names)
        for idx, val in enumerate(file_names):
            # check if h5 is corrupted by checking for file signature exception
            try:
                f = h5py.File(self.root_dir + '/' + file_names[idx], 'r') # ,
                # driver = 'core', backing_store=False)
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

        if self.file_handle_dict.get(file_idx) is None:
            if len(self.file_idx_queue) >= MAX_QUEUE_LENGTH:
                popped_idx = self.file_idx_queue.pop()
                self.file_handle_dict[popped_idx].close()
                del self.file_handle_dict[popped_idx]

            self.file_idx_queue.append(file_idx)
            self.file_handle_dict[file_idx] = h5py.File(self.root_dir + '/' + self.file_names[file_idx], 'r')

        current_file = self.file_handle_dict[file_idx]

        data = current_file['rgb'][idx]
        targets = current_file['targets'][idx]

        # enhance the acceleration data
        if self.raiscar==False:
            targets[self.target_idx['gas']] = targets[self.target_idx['gas']] - targets[self.target_idx['brake']]

        else:
            targets[self.target_idx['steer']] *= -1.0  # for some reason the steering was inverted
            targets[self.target_idx['gas']] = abs(targets[self.target_idx['gas']])
            # also: if in raiscar mode, extract original sized image
            try:
                orig_image = current_file['rgb_original'][idx]
            except KeyError:
                orig_image = np.multiply(current_file['rgb'][idx], 256,
                                         type=np.uint8)


        # when in raiscar mode, return also original image
        if self.transform:
            sample = (self.transform(data),
                      torch.Tensor(targets))
            if self.raiscar:
                sample = (self.transform(data), torch.Tensor(targets), orig_image)
        else:
            sample = (data,
                      targets)
            if self.raiscar:
                sample = (data, targets, orig_image)

        return sample

def better_random_split(dataset_enhanced,
                        dataset_clean,
                        fraction):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths... but better!

    Arguments:
        dataset_enhanced: The dataset with transforms
        dataset_clean: the dataset with only the necessary transforms and in sequential order
        fraction: the amount of data to be split
    """
    assert fraction < 1, "Fraction should be < 1"
    assert len(dataset_enhanced) == len(dataset_clean)

    total_length = len(dataset_enhanced)
    train_length = int(fraction * total_length)
    eval_length = total_length - train_length

    val_idx0 = np.random.randint(train_length)

    train_idx_lst = np.append(np.arange(val_idx0),
                              np.arange(val_idx0 + eval_length, total_length))
    eval_idx_lst = np.arange(val_idx0, val_idx0 + eval_length)

    np.random.shuffle(train_idx_lst)

    return Subset(dataset_enhanced, train_idx_lst), Subset(dataset_clean, eval_idx_lst)


def optimized_split(dataset_enhanced,
                    dataset_clean,
                    fraction,
                    sets_per_file=200):
    assert fraction < 1, "Fraction should be < 1"
    assert len(dataset_enhanced) == len(dataset_clean)

    total_length = len(dataset_enhanced)
    amount_files = int(total_length / sets_per_file)

    train_length = int(fraction * total_length)

    indices = np.arange(total_length).reshape((amount_files,
                                               sets_per_file))

    # shuffle all the sets in the files
    for idx in range(indices.shape[0]):
        indices[idx] = np.random.permutation(indices[idx])

    # shuffle all the files around and flatten the array
    indices = np.random.permutation(indices).flatten()

    return Subset(dataset_enhanced, indices[:train_length]), \
        Subset(dataset_clean, indices[train_length:])


""" Just show pretty, enhanced samples"""
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain')
    parser.add_argument("-r", "--raiscar",
                        help="raiscar enable flag",
                        action="store_true")
    args = parser.parse_args()

    traindata_path = args.train
    raiscar = args.raiscar

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
                          transform=randomized,raiscar=raiscar)

    orig_train_set = H5Dataset(root_dir = traindata_path,
                               transform=un_composed, raiscar=raiscar)

    browser = ImageBrowser([train_set, orig_train_set],raiscar=raiscar)
    browser.show()


if __name__ == "__main__":
    main()
