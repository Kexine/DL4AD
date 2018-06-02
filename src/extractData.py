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

# from ImageHandling import ImageBrowser

import warnings
# Ignore warnings
warnings.filterwarnings("ignore")

DATA_DIR = '../data/AgentHuman'

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
def load_model():
    '''
    Check if a pre trained model exists and load it if found
    '''
    print("Checking if some model exists...")
    model_path = '/home/hive/DL4AD/src/model/model.pt'
    if Path(model_path).is_file():
        model.load_state_dict(torch.load(model_path))
        print("Model was found and loaded!")
    else:
        print("No model found, starting training with new model!")

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

        # print("Idx: {}, Type: {}".format(idx, type(idx)))  # TODO: Remove me
        f = h5py.File(self.root_dir + '/' + self.file_names[file_idx], 'r')

        # for magic idx numers inspect class description
        data = f['rgb']
        targets = f['targets']
        sample = {'filename' : self.file_names[file_idx],
                  'data' : data[idx], 'targets' : targets[idx]}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        return sample


if  __name__=="__main__":

    ### PSEUDO MAIN ###

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


    train_set = H5Dataset(root_dir = DATA_DIR + '/SeqTrain', transform=composed)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=32, shuffle=True, pin_memory=False)

    orig_train_set = H5Dataset(root_dir = DATA_DIR + '/SeqTrain', transform=un_composed)

    browser = ImageBrowser(train_set, orig_train_set)
    browser.show()

    # g = PrimitiveGUI()
    # g(train_set, orig_train_set)

    # if no index given, generate random index pair
    # idx = random.randrange(0,len(train_set))

    # sample = train_set[idx]
    # show_image(sample, True)
    # matplot_display(sample)

    # orig_sample = orig_train_set[idx]
    # # show_image(orig_sample, True)
    # matplot_display(orig_sample)

    # plt.show()



    # print(next(iter(train_loader)))
