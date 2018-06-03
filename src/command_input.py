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

import warnings
torch.manual_seed(1)

# define the cuda device
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

def load_model(model_path):
    '''
    Check if a pre trained model exists and load it if found
    '''
    print("Checking if some model exists...")

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model was found and loaded!")
    else:
        print("No model found, starting training with new model!")


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


# Our neural network
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
        m = m.view(m.shape[0], -1)
        m = self.fc3(m)
        m= self.fc_drop(m)
        m = relu(m)

        m = self.fc4(m)
        m = self.fc_drop(m)
        m = relu(m)
        ####################################

        #########for control################
        c = c.view(c.shape[0], -1)
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

        j = self.fc7(j)
        j = self.fc_drop(j)
        j = relu(j)

        #### output action##########
        action = self.fc8(j)

        return action


def train(epoch, train_loader):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data,
                       target[:,target_idx['speed']],
                       target[:,target_idx['command']])

        # Calculation of the loss function
        output_target = target[:,[target_idx['steer'],
                                  target_idx['gas']]]  # DONE: Remove magic numbers
        loss = nn.MSELoss()(output.double(), output_target.double())

        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            save_model(model, model_path)


if  __name__=="__main__":
    model_path = '../model/model.pt'
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
                                               batch_size=64, # TODO: Decide on batchsize
                                               shuffle=True,
                                               pin_memory=False)

    # orig_train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    model = Net().to(device)
    load_model(model_path)


    relu = F.relu

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # criterion = nn.CrossEntropyLoss()

    lossx = []
    num_train_epochs = 1
    for epoch in range(1, num_train_epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move the input and target data on the GPU
            data, target = data.to(device), target.to(device)
            # Zero out gradients from previous step
            optimizer.zero_grad()
            # Forward pass of the neural net
            output = model(data,
                           target[:,SPEED_IDX],
                           target[:,HIGH_LEVEL_COMMAND_IDX])
            # Calculation of the loss function
            output_target = target[:,:2]  # TODO: remove magic numbers
            loss = nn.MSELoss()(output.double(), output_target.double())
            # Backward pass (gradient computation)
            loss.backward()
            # Adjusting the parameters according to the loss function
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            save_model(model, model_path)

    # browser = ImageBrowser(train_set, orig_train_set)
    # browser.show()
