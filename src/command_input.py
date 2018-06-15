#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 1
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

from __future__ import print_function, division
import h5py
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
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

# our custom modules
from customTransforms import *
from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx, better_random_split, optimized_split
from CustomLoss import WeightedMSELoss

import sys
import time

import warnings
torch.manual_seed(1)

import cProfile, pstats, io

try:
    import progressbar
except ModuleNotFoundError:
    progressbar = None


# define the cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ignore warnings
warnings.filterwarnings("ignore")

### CONSTANTS ###

COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}

def load_model(model, model_path):
    '''
    Check if a pre trained model exists and load it if found
    '''
    print("Checking if some model exists...")

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model was found and loaded!")
    else:
        print("No model found, starting training with new model!")


def save_model(model, model_path, epoch, train_loss = None):
    torch.save(model.state_dict(), model_path)
    # also store a csv file with the train loss
    if train_loss is not None:
        csv_path = model_path.replace("pt", "csv")
        df = pd.DataFrame({'col1':[int(epoch)], 'col2':[train_loss]})
        with open(csv_path, 'a') as f:
            df.to_csv(f,
                      sep="\t",
                      header=False,
                      index=False)


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
        x = F.relu(x)

        x= self.conv2(x)
        x= self.conv2_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv3(x)
        x= self.conv3_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv4(x)
        x= self.conv4_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv5(x)
        x= self.conv5_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv6(x)
        x= self.conv6_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv7(x)
        x= self.conv7_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        x= self.conv8(x)
        x= self.conv8_bn(x)
        x= self.conv_drop(x)
        x = F.relu(x)

        ###################################

        # x = x.view(-1, 204*92*256)      ### TODO: change this
        x = x.view(-1, 25*11*256)

        #########fully connected layers####
        x = self.fc1(x)
        x = self.fc_drop(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc_drop(x)
        x = F.relu(x)
        ####################################

        ####for  measurement(speed)#########
        m = m.view(m.shape[0], -1)
        m = self.fc3(m)
        m= self.fc_drop(m)
        m = F.relu(m)

        m = self.fc4(m)
        m = self.fc_drop(m)
        m = F.relu(m)
        ####################################

        #########for control################
        c = c.view(c.shape[0], -1)
        c = self.fc3(c)
        c = self.fc4(c)

        ###concatenating previous layers####
        j = torch.cat((x,m,c), 1)
        j = self.fc5(j)
        j = self.fc_drop(j)
        j = F.relu(j)

        ####################################
        j = self.fc6(j)
        j= self.fc_drop(j)
        j = F.relu(j)

        j = self.fc7(j)
        j = self.fc_drop(j)
        j = F.relu(j)

        #### output action##########
        action = self.fc8(j)

        return action


def evaluate(model,
             eval_loader,
             loss_function,
             weights,
             verbose=True):
    with torch.no_grad():
        loss = 0
        for eval_idx, (data, target) in enumerate(eval_loader):
            print("\rEvaluation in progress {:.0f}%/100%".format((eval_idx+1)/len(eval_loader)*100),
                  end="",
                  flush=True)
            data, target = data.to(device), target.to(device)
            output = model(data,
                           target[:,target_idx['speed']],
                           target[:,target_idx['command']])
            output_target = target[:,[target_idx['steer'],
                                      target_idx['gas']]]
            loss += loss_function(output.double(),
                                  output_target.double(),
                                  weights.double()).item()

    avg_loss = loss/len(eval_loader)
    return avg_loss


def main():
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="A (existing?) model file to store to",
                        default='../model/command_input.pt')
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain')
    parser.add_argument("-e", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=200,
                        type=int)
    parser.add_argument("-v", "--valfraction",
                        help="Fraction of training/validation part of data set",
                        default=0.9,
                        type=float)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=200,
                        type=int)

    args = parser.parse_args()

    model_path = args.model
    traindata_path = args.train
    eval_rate = args.evalrate
    eval_fraction = args.valfraction
    batch_size = args.batchsize

    composed = RandomApplyFromList([ContrastNBrightness(1.5,0.5),
                                    GaussianBlur(1.5),
                                    SaltNPepper(0.1),
                                    GaussianNoise(0, 0.1),
                                    RegionDropout((10, 10),10)],
                                   normalize = True,
                                   std=1)
    un_composed = transforms.Compose([JustNormalize(std=1)])

    train_set = H5Dataset(root_dir = traindata_path,
                          transform=composed)
    eval_set = H5Dataset(root_dir = traindata_path,
                         transform=un_composed)

    # orig_train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    model = Net().to(device)
    load_model(model, model_path)

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # criterion = nn.CrossEntropyLoss()

    ############### Training
    lossx = []
    num_train_epochs = 5

    weights = torch.eye(2)
    weights[0,0] = 0.7
    weights[1,1] = 0.3  # this is the strange lambda
    weights = weights.to(device)

    loss_function = WeightedMSELoss()

    start_time = time.time()


    # show loss each after m batches
    BATCH_LOSS_RATE = 1

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, # TODO: Decide on batchsize
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=8)

    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    for epoch in range(1, num_train_epochs + 1):
        print("---------------------------------------------------------------")
        print("EPOCH {}".format(epoch))
        print("Batch Size: {}\t| Eval Rate: {}".format(batch_size, eval_rate))
        print("Splitted Training Set into {} Training and {} Validation:".format(eval_fraction,round(1-eval_fraction,2)))
        print("{} Training Samples\t| {} Evaluation Samples".format(len(train_set), len(eval_set)))
        print("{} Training Batches\t| {} Evaluation Batches".format(len(train_loader), len(eval_loader)))
        print("---------------------------------------------------------------")
        model.train()

        try:
            if progressbar is not None:
                bar = progressbar.ProgressBar(max_value = len(train_loader))
                bar.dynamic_messages['Loss'] = float('inf')
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move the input and target data on the GPU
                data = data.to(device)
                target = target.to(device)
                # Zero out gradients from previous step
                optimizer.zero_grad()
                # Forward pass of the neural net
                output = model(data,
                               target[:,target_idx['speed']],
                               target[:,target_idx['command']])

                # Calculation of the loss function
                output_target = target[:,[target_idx['steer'],
                                          target_idx['gas']]]  # DONE: remove magic numbers

                loss = loss_function(output.double(),
                                     output_target.double(),
                                     weights.double())
                # Backward pass (gradient computation)
                loss.backward()
                # Adjusting the parameters according to the loss function
                optimizer.step()
                if batch_idx % BATCH_LOSS_RATE  == 0 and batch_idx != 0:
                    print('{:04.2f}s - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        time.time() - start_time,
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                    if progressbar is not None:
                        bar.update(batch_idx, 'Loss', loss.item())

                    save_model(model, model_path, epoch,
                               train_loss = loss.item())

                # ---------- Validation after n batches
                if batch_idx % eval_rate == 0 and batch_idx != 0:
                    print("---------------------------------------------------------------")
                    model.eval()
                    avg_loss = evaluate(model,
                                        eval_loader,
                                        loss_function,
                                        weights)
                    print("\n{:04.2f}s - Average Evaluation Loss: {:.6f}".format(time.time() - start_time,
                                                                                 avg_loss))
                    print("---------------------------------------------------------------")
                    csv_path_eval_loss = model_path.replace('.pt', '_evalloss.csv')
                    if avg_loss is not None:
                        df_eval_loss = pd.DataFrame({'col1':[int(epoch)], 'col2':[avg_loss]})

                        with open(csv_path_eval_loss, 'a') as f:
                            df_eval_loss.to_csv(f,
                                                sep="\t",
                                                header=False,
                                                index=False)

        except KeyboardInterrupt:
            print("Abort detected! Saving the model and exiting (Please don't hit C-c again >.<)")
            save_model(model, model_path, epoch,
                       train_loss = loss.item())
            return

        save_model(model, model_path, epoch,
                   train_loss = loss.item())


if  __name__=="__main__":
    main()
