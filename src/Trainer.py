#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 1
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

# basic python stuff
import argparse
import pandas as pd
import os

# pytorch stuff
import torch
import torch.optim as optim
from torchvision import transforms

# custom stuff
from Extractor import H5Dataset
from customTransforms import *
from CustomLoss import WeightedMSELoss

import command_input
import Branched

net_types = ['command_input',
             'branched',
             'command_input_raiscar',
             'branched_raiscar']

def load_model(model, model_path):
    '''
    Check if a pre trained model exists and load it if found
    '''
    print("Checking if some model exists... ", end="")

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model was found and loaded!")
    else:
        print("No model found, starting training with new model!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("net_type",
                        help="What kind of Net do you want to train?\n" + \
                        "|".join(net_types))
    parser.add_argument("-m", "--model",
                        help="A (existing?) model file to store to",
                        default='../model/command_input.pt')
    parser.add_argument("-t", "--train",
                        help="Directory of the train data",
                        default='../data/AgentHuman/SeqTrain/train')
    parser.add_argument("-v", "--val",
                        help="Directory of the validation data",
                        default='../data/AgentHuman/SeqTrain/val')
    parser.add_argument("-r", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=263,  # this is basically 10 evals each epoch
                        type=int)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=200,
                        type=int)
    parser.add_argument("-e", "--epochs",
                        help="Train for how many epochs? (15)",
                        type=int,
                        default = 15)
    parser.add_argument("--no_transforms",
                        help="Don't apply any transforms on the test set.",
                        action='store_true')


    args = parser.parse_args()

    net_type = args.net_type
    model_path = args.model
    traindata_path = args.train
    valdata_path = args.val
    eval_rate = args.evalrate
    batch_size = args.batchsize
    no_transforms = args.no_transforms
    amount_epochs = args.epochs

    assert net_type in net_types, "Please choose a proper Net!"

    # -------------------- Get the cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    # -------------------- Prepare the datasets
    basic_transforms = transforms.Compose([JustNormalize(std=1)])  # TODO: maybe we don't need the Compose()?

    # the transformations with the magnitudes to apply to the train data
    if no_transforms:
        train_transforms = basic_transforms
    else:
        train_transforms = RandomApplyFromList([ContrastNBrightness(1.5,0.5),
                                                GaussianBlur(1.5),
                                                SaltNPepper(0.1),
                                                GaussianNoise(0, 0.1),
                                                RegionDropout((10, 10),10)],
                                               normalize = True,
                                               std=1)

    train_set = H5Dataset(root_dir = traindata_path,
                          transform = train_transforms)
    eval_set = H5Dataset(root_dir = valdata_path,
                         transform = basic_transforms)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=4,
                                               drop_last=True)

    eval_loader = torch.utils.data.DataLoader(eval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=4,
                                              drop_last=True)

    # -------------------- Prepare the model

    if net_type == 'command_input':
        model = command_input.Net().to(device)
    elif net_type == 'branched':
        model = command_input.Net().to(device)
    elif net_type == 'command_input_raiscar':
        raise NotImplementedError
    elif net_type == 'branched_raiscar':
        raise NotImplementedError

    load_model(model, model_path)

    # -------------------- Prepare the optimizer + loss function

    # define the weights
    weights = torch.eye(2)
    weights[0,0] = 0.75
    weights[1,1] = 0.25

    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    loss_function = WeightedMSELoss()

    # -------------------- Prepare the dataframe for logging the loss
    loss_df = pd.DataFrame([], columns=['train_loss', 'eval_loss', 'epoch'])
