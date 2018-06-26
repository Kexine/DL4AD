#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Learning in Autonomous Driving
Project 4: Exercise sheet 3, Task 1
Michael Floßmann, Kshitij Sirohi, Hendrik Vloet
"""

# basic python stuff
import argparse

# pytorch stuff
import torch
from torchvision import transforms

# custom stuff
from customTransforms import *


net_types = ['command_input',
             'branched',
             'command_input_raiscar',
             'branched_raiscar']

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
    parser.add_argument("-e", "--evalrate",
                        help="Evaluate every [N] training batches",
                        default=263,  # this is basically 10 evals each epoch
                        type=int)
    parser.add_argument("-b", "--batchsize",
                        help="Size of batches",
                        default=200,
                        type=int)


    args = parser.parse_args()

    net_type = args.net_type
    model_path = args.model
    traindata_path = args.train
    valdata_path = args.val
    eval_rate = args.evalrate
    batch_size = args.batchsize

    assert net_type in net_types, "Please choose a proper Net!"

    # the transformations with the magnitudes to apply to the train data
    train_transforms = RandomApplyFromList([ContrastNBrightness(1.5,0.5),
                                            GaussianBlur(1.5),
                                            SaltNPepper(0.1),
                                            GaussianNoise(0, 0.1),
                                            RegionDropout((10, 10),10)],
                                           normalize = True,
                                           std=1)

    basic_transforms = transforms.Compose([JustNormalize(std=1)])  # TODO: maybe we don't need the Compose()?
