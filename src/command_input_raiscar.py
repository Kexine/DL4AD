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
from Extractor import H5Dataset, target_idx_raiscar, better_random_split, optimized_split
from CustomLoss import WeightedMSELoss

import sys
import time

import warnings
torch.manual_seed(1)

import cProfile, pstats, io

try:
    import progressbar
    progress_widgets = [progressbar.widgets.DynamicMessage('loss'),
                        ' ', progressbar.widgets.Percentage(),
                        ' ', progressbar.widgets.Bar(),
                        ' ', progressbar.widgets.Timer(),
                        ' ', progressbar.widgets.AdaptiveETA(samples = 200),
                        ' ', progressbar.widgets.CurrentTime()]
except ModuleNotFoundError:
    progressbar = None


# define the cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

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
        # self.fc5= nn.Linear(768,512)
        self.fc5= nn.Linear(640,512)
        self.fc6= nn.Linear(512,256)
        self.fc7= nn.Linear(256,256)

        #5 for action output
        self.fc8= nn.Linear(256,2)


    def forward(self, x,c):
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

        # ####for  measurement(speed)#########
        # m = m.view(m.shape[0], -1)
        # m = self.fc3(m)
        # m= self.fc_drop(m)
        # m = F.relu(m)
        #
        # m = self.fc4(m)
        # m = self.fc_drop(m)
        # m = F.relu(m)
        # ####################################

        #########for control################
        c = c.view(c.shape[0], -1)
        c = self.fc3(c)
        c = self.fc4(c)

        ###concatenating previous layers####
        j = torch.cat((x,c), 1)
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
        model = model.to(device)

        if progressbar is not None:
            eval_bar = progressbar.ProgressBar(max_value = len(eval_loader),
                                               widgets = progress_widgets)

        for eval_idx, (data, target) in enumerate(eval_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data,
                            target[:,target_idx_raiscar['command']])
            output_target = target[:,[target_idx_raiscar['steer'],
                                      target_idx_raiscar['gas'],
                                      ]]
            # output_target[:,1] = output_target[:,1] - target[:,target_idx_raiscar['brake']]
            current_loss = loss_function(output.double(),
                                         output_target.double(),
                                         weights.double()).item()
            loss += current_loss

            if progressbar is not None:
                eval_bar.update(eval_idx, loss=loss/(eval_idx + 1))
            else:
                print("\rEvaluation in progress {:.0f}%/100%".format((eval_idx+1)/len(eval_loader)*100),
                      end="",
                      flush=True)

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
                        default='../data/AgentHuman/SeqTrain/train')
    parser.add_argument("-v", "--val",
                        help="Directory of the validation data",
                        default='../data/AgentHuman/SeqTrain/val')
    parser.add_argument("-e", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=263,  # this is basically 10 evals each epoch
                        type=int)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=200,
                        type=int)

    args = parser.parse_args()

    model_path = args.model
    traindata_path = args.train
    valdata_path = args.val
    eval_rate = args.evalrate
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
                          transform=un_composed)
    eval_set = H5Dataset(root_dir = valdata_path,
                         transform=un_composed)
    # orig_train_set = H5Dataset(root_dir = '../data/AgentHuman/SeqTrain', transform=un_composed)

    model = Net().to(device)
    load_model(model, model_path)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    lossx = []
    weights = torch.eye(2)
    weights[0,0] = 0.75
    weights[1,1] = 0.25
    weights = weights.to(device)

    loss_function = WeightedMSELoss()

    num_train_epochs = 15

    start_time = time.time()

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size, # TODO: Decide on batchsize
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=4,
                                               drop_last=True)

    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)

    loss_df = pd.DataFrame([], columns=['train_loss', 'eval_loss', 'epoch'])
    ############### Training
    model.train()

    for epoch in range(1, num_train_epochs + 1):
        print("---------------------------------------------------------------")
        print("EPOCH {}".format(epoch))
        print("Batch Size: {}\t| Eval Rate: {}".format(batch_size, eval_rate))
        print("{} Training Samples\t| {} Evaluation Samples".format(len(train_set), len(eval_set)))
        print("{} Training Batches\t| {} Evaluation Batches".format(len(train_loader), len(eval_loader)))
        print("---------------------------------------------------------------")
        try:
            if progressbar is not None:
                bar = progressbar.ProgressBar(max_value = len(train_loader),
                                              widgets = progress_widgets)

            # initialize the train loss storing array
            train_loss = 0
            amount_trains = 0
            # -------------------- Actual training
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move the input and target data on the GPU
                data = data.to(device)
                target = target.to(device)
                # Zero out gradients from previous step
                optimizer.zero_grad()
                # Forward pass of the neural net
                output = model(data,
                                target[:,target_idx_raiscar['command']])

                output_target = target[:,[target_idx_raiscar['steer'],
                                          target_idx_raiscar['gas']]]
                                            # DONE: remove magic numbers
                # output_target[:,1] = output_target[:,1] - target[:,target_idx_raiscar['brake']]



                loss = loss_function(output.double(),
                                     output_target.double(),
                                     weights.double())
                # Backward pass (gradient computation)
                loss.backward()
                # Adjusting the parameters according to the loss function
                optimizer.step()

                # store the training loss
                train_loss += loss.item()
                amount_trains += 1

                if progressbar is not None:
                    bar.update(batch_idx, loss=train_loss / (amount_trains))
                else:
                    print('{:04.2f}s - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        time.time() - start_time,
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                if batch_idx % eval_rate  == eval_rate - 1:
                    # ---------- Validation
                    print("Evaluation ----------------------------------------------------")
                    model.eval()
                    eval_loss = evaluate(model,
                                         eval_loader,
                                         loss_function, weights)
                    print("\n{:04.2f}s - Average Evaluation Loss: {:.6f}".format(time.time() - start_time,
                                                                                 eval_loss))
                    print("---------------------------------------------------------------")

                    model.train()
                    loss_df = loss_df.append(pd.DataFrame([[train_loss/amount_trains, eval_loss, epoch]],
                                                          columns=['train_loss', 'eval_loss', 'epoch']),
                                             ignore_index=True)

                    train_loss = 0
                    amount_trains = 0

                    # # ---------- Also, save the model here
                    # save_model(model, model_path)


        except KeyboardInterrupt:
            print("Abort detected! Saving the model and exiting (Please don't hit C-c again >.<)")
            break

        save_model(model, model_path)

        with open(model_path.replace(".pt", "_loss.csv"), 'w') as f:
            loss_df.to_csv(f, sep="\t", header=True, index=True)
            # TODO: can also be done with appending instead of overwriting


if  __name__=="__main__":
    main()
