#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 16:22:39 2018

@author: sirohik
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

from FallbackGUI import PrimitiveGUI
from ImageHandling import ImageBrowser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# from ImageHandling import ImageBrowser

import warnings
torch.manual_seed(1)
device = torch.device("cuda")
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


def matplot_display(sample):
    """
    gets an BGR image, converts it to RGB and displays it with matplotlib
    maybe easier to handle than opencv?
    """
    img = sample['data'].numpy().transpose((1,2,0))
    plt.figure()
    plt.title("File: {}".format(sample['filename']))
    # converg BGR to RGB
    rgb = img[...,::-1]
    plt.imshow(rgb)

def show_image(sample, trans_en=False):
    '''
    sample: sampled image
    '''

    if trans_en:
        img = sample['data'].numpy().transpose((1,2,0))
    else:
        img = sample['data']

    # magic 24 is the position of the high level command in the target array
    # print(COMMAND_DICT[int(sample['targets'][HIGH_LEVEL_COMMAND_IDX])])
    high_level_command = COMMAND_DICT[int(sample['targets'][HIGH_LEVEL_COMMAND_IDX])]

    # magic 0 is the position of the high level command in the target array
    steering_angle = sample['targets'][STEERING_ANGLE_IDX]

    height, width = img.shape[:2]
    # show image with bigger resolution, does not affect the actual data
    res = cv2.resize(img,(4*width, 5*height), interpolation = cv2.INTER_CUBIC)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(res,'Command: {}'.format(high_level_command),(5,15 ), font, 0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(res,'Steering: {:.5f}'.format(steering_angle),(5,30 ), font, 0.5,(0,0,255),1,cv2.LINE_AA)

    img_title = "File: {}| Command: {}| Steering Angle: {:.5f}".format(sample['filename'],
                                                                       high_level_command,
                                                                       steering_angle)
    cv2.imshow(img_title,
               res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        # print(self.file_names)
        # print(len(self.file_names))
        self.file_idx = 0

    def __len__(self):
        return len(self.file_names)*IMAGES_PER_FILE

    def __getitem__(self, idx):
        self.file_names =  sorted(os.listdir(self.root_dir))

        # The input idx seems sequential but we're actually going through the
        # images in each file and going through all the files in order.
        # Note: Danger! This >>WILL<< break, if any h5 file doesn't have
        # >>EXACTLY<< 200 images
        raw_idx = idx
        file_idx = int(idx / IMAGES_PER_FILE)
        idx = idx % IMAGES_PER_FILE

        f = h5py.File(self.root_dir + '/' + self.file_names[file_idx], 'r')

        # for magic idx numers inspect class description
        data = f['rgb']
        targets = f['targets']
        sample = (data[idx],
                  targets[idx])

        if self.transform:
            sample = (self.transform(data[idx]),
                      torch.Tensor(targets[idx]))
        else:
            sample = (data[idx],
                      targets[idx])

        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #8 layers if con layers for image module
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=0,stride =2) #(output = 100x44)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)#(output = 100x44)
        self.conv3= nn.Conv2d(32, 64, kernel_size=3, padding=1,stride =2) #(output = 50x22)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#(output = 50x22)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1,stride =2)#(output = 25x11)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#(output = 25*11)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)#(output = 25*11)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)#(output = 25*11)

        #defining 2 different dropouts required after the conv and fc layers
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc_drop = nn.Dropout2d(p=0.5)


        #batch normalisers for every convolution layer
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv7_bn = nn.BatchNorm2d(256)
        self.conv8_bn = nn.BatchNorm2d(256)


        #2 fc layers for image module
        # self.fc1 = nn.Linear(204*92*256, 512) #(please reconfirm with team)
        self.fc1 = nn.Linear(25*11*256, 512)
        self.fc2 = nn.Linear(512, 512)

        #3 fc layers for control and measurement modules
        self.fc3= nn.Linear(1,128)
        self.fc4= nn.Linear(128,128)

        #4 fc layers for concatenated module
        self.fc5= nn.Linear(768,512)
        self.fc6= nn.Linear(512,256)
        self.fc7= nn.Linear(256,256)

        #5 for action output
        self.fc8= nn.Linear(256,2)


    def forward(self, x,m,c):

        #######conv layers##############
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv_drop(x)
        x = relu(x)

        x= self.conv2(x)
        x= self.conv2_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv3(x)
        x= self.conv3_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv4(x)
        x= self.conv4_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv5(x)
        x= self.conv5_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv6(x)
        x= self.conv6_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv7(x)
        x= self.conv7_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        x= self.conv8(x)
        x= self.conv8_bn(x)
        x= self.conv_drop(x)
        x = relu(x)

        ###################################

        # x = x.view(-1, 204*92*256)      ### TODO: change this
        x = x.view(-1, 25*11*256)

        #########fully connected layers####
        x = self.fc1(x)
        x = self.fc_drop(x)
        x = relu(x)

        x = self.fc2(x)
        x = self.fc_drop(x)
        x = relu(x)


        ####################################

        ####for  measurement(speed)#########
        m = self.fc3(m)
        x= self.fc_drop(x)
        x = relu(x)

        m = self.fc4(m)
        m = self.fc_drop(m)
        m = relu(m)
        ####################################

        #########for control################
        c = self.fc3(c)
        c = self.fc4(c)

        ###concatenating previous layers####
        j = torch.cat((x,m,c), 1)
        j = self.fc5(j)
        j = self.fc_drop(j)
        j = relu(j)

        ####################################
        j = self.fc6(j)
        j= self.fc_drop(j)
        j = relu(j)

        j = torch.cat((x,m,c), 1)
        j = self.fc7(j)
        j = self.fc_drop(j)
        j = relu(j)

        #### output action##########
        action = self.fc8(j)

        return action


def train(epoch, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print('type data: {}, type target: {}'.format(type(data), type(target)))
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data,
                       target[SPEED_IDX],
                       target[HIGH_LEVEL_COMMAND_IDX])
        # Calculation of the loss function
        loss = nn.CrossEntropyLoss(output, target[0:1])
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))




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
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=64,
                                               shuffle=True,
                                               pin_memory=False)

    orig_train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    model = Net().to(device)

    relu = F.relu

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # criterion = nn.CrossEntropyLoss()

    lossx = []

    num_train_epochs = 1
    for epoch in range(1, num_train_epochs + 1):
        train(epoch, train_loader)

    # browser = ImageBrowser(train_set, orig_train_set)
    # browser.show()
