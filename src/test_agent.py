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
import pandas as pd
import os, os.path
import random

from FallbackGUI import PrimitiveGUI
from ImageHandling import ImageBrowser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# our custom modules
from customTransforms import *
from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx
from command_input import Net

import argparse

import warnings
torch.manual_seed(1)

BATCH_SIZE = 32

def load_model(model_path):
    '''
    Check if a pre trained model exists and load it if found
    '''

    model.load_state_dict(torch.load(model_path))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if  __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test",
                        help="Directory of the test data",
                        default='../data/AgentHuman/SeqVal')
    parser.add_argument("-m", "--model",
                        help="Location of the model to use",
                        default='../model/command_input.pt')

    args = parser.parse_args()

    testdata_path = args.test
    model_path = args.model
    transforms = transforms.Compose([transforms.ToTensor()])

    csv_path = "../test_loss/loss.csv"
    test_set = H5Dataset(root_dir = testdata_path,
                          transform=transforms)


    test_loader = torch.utils.data.DataLoader(test_set,
                                           batch_size=BATCH_SIZE, # TODO: Decide on batchsize
                                           shuffle=True,
                                           pin_memory=False)

    model = Net().to(device)

    model.load_state_dict(torch.load(model_path))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print(len(test_loader))
        for idx ,(data, target) in enumerate(test_loader):
            # print(idx)
            data, target = data.to(device), target.to(device)
            output = model(data,
                           target[:,target_idx['speed']],
                           target[:,target_idx['command']])
            output_target = target[:,[target_idx['steer'], target_idx['gas']]]

            test_loss += F.mse_loss(output, output_target, size_average=False).item() # sum up batch loss
            # print("Prediction of data: {}".format(output))
            # print("Prediction of targets: {}".format(output_target))
            if idx%100 == 0:
                test_loss /= len(test_loader.dataset)
                df = pd.DataFrame([test_loss])
                with open(csv_path, 'a') as f:
                    df.to_csv(f,
                              sep="\t",
                              header=False,
                              index=False)
                print(test_loss)
