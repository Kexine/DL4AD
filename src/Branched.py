#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 2
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

#!/usr/bin/env python3
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
from torchvision import datasets, transforms

# our custom modules
from customTransforms import *
from ImageHandling import ImageBrowser
from Extractor import H5Dataset, target_idx, better_random_split, optimized_split
from CustomLoss import WeightedMSELoss

from command_input import load_model, save_model, progress_widgets

import warnings
torch.manual_seed(1)

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

BATCH_LOSS_RATE = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ignore warnings
warnings.filterwarnings("ignore")

COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}


class Net(nn.Module):
    branches = []
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
        # layer before branching
        self.fc5= nn.Linear(640,512)

        # submodules for each branch
        self.fc6_branch1
        for branch in COMMAND_DICT.keys():
            self.branch_submodules[branch] = nn.Linear(512, 256)

        # self.fc6= nn.Linear(512,256)
        # self.fc7= nn.Linear(256,256)
        # self.fc8= nn.Linear(256,2)


    def forward(self, x, speed, command):
        batch_size = x.shape[0]

        #######conv layers##############
        x= self.conv1(x)
        x= self.conv1_bn(x)
        x= self.conv_drop(x)
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

        x = x.view(-1, 25*11*256)  # TODO: please explain this comment: "do change this"

        #########fully connected layers####
        x = self.fc1(x)
        x= self.fc_drop(x)
        x = F.relu(x)

        x = self.fc2(x)
        x= self.fc_drop(x)
        x = F.relu(x)

        #####do something for control########

        ####for  measurement(speed)#########
        ###not to use in real raiscar#######
        speed = speed.view(speed.shape[0], -1)
        speed = self.fc3(speed)
        speed= self.fc_drop(speed)
        speed = F.relu(speed)

        speed = self.fc4(speed)
        speed= self.fc_drop(speed)
        speed = F.relu(speed)
        ####################################

        ####################################
        j = torch.cat((x,speed), 1)
        j = self.fc5(j)
        j = self.fc_drop(j)
        j = F.relu(j)

        # -------------------- applying branch submodules

        # apply to each branch
        command_np = command.cpu().numpy()
        output = torch.zeros(batch_size)
        for branch in COMMAND_DICT.keys():
            command_mapping = np.where(command == branch)
            branch_output = j[command_mapping,:].view(-1,512)

            branch_output = self.branch_submodules[branch](branch_output)

            # output[command_mapping] =
            # self.f6_1(branch_input)
        exit()


        for i in range(0, len(branch_config)):
            branch_output = self.fc6(j)
            branch_output= self.fc_drop(branch_output)
            branch_output = F.relu(branch_output)
            branch_output = self.fc7(branch_output)
            branch_output= self.fc_drop(branch_output)
            branch_output = F.relu(branch_output)
            branches.append(self.fc8(branch_output))
        #have to look for this regarding the dataset , on how to use it?

        return branches

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
            data, target = data.to(device), target.to(device)
            output_branches = model(data,
                                    target[:, target_idx['speed']],
                                    target[:, target_idx['command']])

            # Calculation of the loss function
            print("len(target[0].shape): {}".len(target[0].shape))
            exit()
            for c in range(0,len(target[0].shape)):
                output_target = target[:,[target_idx['steer'],
                                      target_idx['gas']]]
                current_branch = int(target[c,target_idx['command']] - 2)
                output = output_branches[current_branch]

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
    # --------------- Parse arguments ---------------
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="A (existing?) model file to store to",
                        default='../model/branched.pt')
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain')
    parser.add_argument("-v", "--val",
                        help="Directory of the validation data",
                        default='../data/AgentHuman/SeqTrain/val')
    parser.add_argument("-e", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=200,
                        type=int)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=100,
                        type=int)

    args = parser.parse_args()

    model_path = args.model
    traindata_path = args.train
    valdata_path = args.val
    eval_rate = args.evalrate
    batch_size = args.batchsize

    # --------------- Prepare Datasets ---------------
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

    eval_set = H5Dataset(root_dir = valdata_path,
                         transform=un_composed)


    # --------------- Init training ---------------
    model = Net().to(device)
    load_model(model, model_path)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    lossx = []
    weights = torch.eye(2)
    weights[0,0] = 0.7
    weights[1,1] = 0.3  # this is the strange lambda
    weights = weights.to(device)

    loss_function = WeightedMSELoss(weights.to(device))

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
    # --------------- Train ---------------
    for epoch in range(1, num_train_epochs + 1):
        print("---------------------------------------------------------------")
        print("EPOCH {}".format(epoch))
        print("Batch Size: {}\t| Eval Rate: {}".format(batch_size, eval_rate))
        print("{} Training Samples\t| {} Evaluation Samples".format(len(train_set), len(eval_set)))
        print("{} Training Batches\t| {} Evaluation Batches".format(len(train_loader), len(eval_loader)))
        print("---------------------------------------------------------------")
        model.train()

        try:
            if progressbar is not None:
                bar = progressbar.ProgressBar(max_value = len(train_loader),
                                              widgets = progress_widgets)
            # initialize the train loss storing array
            train_loss = np.zeros((eval_rate,))

            # -------------------- Actual training
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move the input and target data on the GPU
                data, target = data.to(device), target.to(device)

               # Zero out gradients from previous step
                optimizer.zero_grad()
                # Forward pass of the neural net
                output_branches = model(data, target[:, target_idx['speed']], target[:, target_idx['command']])

                # Calculation of the loss function
                train_loss[batch_idx % eval_rate] = 0
                for c in range(0,len(target[0].shape)):
                    print("Foobar! {}".format(c))
                    output_target = target[:,[target_idx['steer'],
                                              target_idx['gas']]]

                    # for command = 2 => output = branches[0] ...
                    current_branch = int(target[c, target_idx['command']] - 2)
                    output = output_branches[current_branch]

                    loss = loss_function(output.double(),
                                         output_target.double())
                    # Backward pass (gradient computation)
                    loss.backward()
                    # Adjusting the parameters according to the loss function
                    optimizer.step()

                    # store the training loss
                    train_loss[batch_idx % eval_rate] += loss.item()

                if progressbar is not None:
                    bar.update(batch_idx, loss=np.mean(train_loss[:batch_idx % eval_rate]))
                else:
                    print('{:04.2f}s - Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        time.time() - start_time,
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

                # ---------- Validation after n batches
                if batch_idx % eval_rate == eval_rate - 1:
                    print("Evaluation ----------------------------------------------------")
                    model.eval()
                    eval_loss = evaluate(model,
                                         eval_loader,
                                         loss_function,
                                         weights)
                    print("\n{:04.2f}s - Average Evaluation Loss: {:.6f}".format(time.time() - start_time,
                                                                                 eval_loss))
                    print("---------------------------------------------------------------")
                    model.train()

                    loss_df = loss_df.append(pd.DataFrame([[np.mean(train_loss), eval_loss, epoch]],
                                                          columns=['train_loss', 'eval_loss', 'epoch']),
                                             ignore_index=True)

        except KeyboardInterrupt:
            print("Abort detected! Saving the model and exiting (Please don't hit C-c again >.<)")
            break

        save_model(model, model_path)

        with open(model_path.replace(".pt", "_loss.csv"), 'w') as f:
            loss_df.to_csv(f, sep="\t", header=True, index=True)
            # TODO: can also be done with appending instead of overwriting


if  __name__=="__main__":
    main()
