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

import warnings
# Ignore warnings
warnings.filterwarnings("ignore")


'''
# Target Array
# 1. Steer, float
# 2. Gas, float
# 3. Brake, float
# 4. Hand Brake, boolean
# 5. Reverse Gear, boolean
# 6. Steer Noise, float
# 7. Gas Noise, float
# 8. Brake Noise, float
# 9. Position X, float
# 10. Position Y, float
# 11. Speed, float
# 12. Collision Other, float
# 13. Collision Pedestrian, float
# 14. Collision Car, float
# 15. Opposite Lane Inter, float
# 16. Sidewalk Intersect, float
# 17. Acceleration X,float
# 18. Acceleration Y, float
# 19. Acceleration Z, float
# 20. Platform time, float
# 21. Game Time, float
# 22. Orientation X, float
# 23. Orientation Y, float
# 24. Orientation Z, float
# 25. High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)
# 26. Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) )
# 27. Camera (Which camera was used)
# 28. Angle (The yaw angle for this camera)
'''

### CONSTANTS ###

HI_LVL_CMD_IDX = 24
STEERING_ANGLE_IDX = 0
COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}



def show_image(sample, trans_en=False):
    '''
    sample: sampled image
    '''

    if trans_en:
        img = sample['data'].numpy().transpose((1,2,0))
    else:
        img = sample['data']

    # magic 24 is the position of the high level command in the target array
    print(COMMAND_DICT[int(sample['targets'][HI_LVL_CMD_IDX])])
    high_level_command = COMMAND_DICT[int(sample['targets'][HI_LVL_CMD_IDX])]

    # magic 0 is the position of the high level command in the target array
    steering_angle = sample['targets'][STEERING_ANGLE_IDX]

    height, width = img.shape[:2]

    # show image with bigger resolution, does not affect the actual data
    res = cv2.resize(img,(4*width, 5*height), interpolation = cv2.INTER_CUBIC)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(res,'Command: {}'.format(high_level_command),(5,15 ), font, 0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(res,'Steering: {:.5f}'.format(steering_angle),(5,30 ), font, 0.5,(0,0,255),1,cv2.LINE_AA)

    cv2.imshow("File: {}| Command: {}| Steering Angle: {:.5f}"
    .format(sample['filename'],  high_level_command,steering_angle),res)
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

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        file_names =  sorted(os.listdir(self.root_dir))

        #if isinstance(idx, tuple):
        #    idx = idx[0]

        print("Idx: {}, Type: {}".format(idx, type(idx)))
        f = h5py.File(self.root_dir + '/' + file_names[idx], 'r')

        # for magic idx numers inspect class description
        data = f['rgb']
        targets = f['targets']
        sample = {'filename' : file_names[idx],
                     'data' : data[idx[-1]], 'targets' : targets[idx[-1]]}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        return sample




if  __name__=="__main__":

    ### PSEUDO MAIN ###

    # dummy composition for debugging
    composed = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    # composed = None

    train_set = H5Dataset(root_dir = 'AgentHuman/SeqTrain', transform=composed)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=32, shuffle=True, pin_memory=False)

    # if no index given, generate random index pair
    file_idx, image_idx = random.randrange(0,len(train_set)), random.randrange(0,200)


    sample = train_set[file_idx,image_idx]
    # show_image(sample, not False)


    # TODO: dataset so anpassen, dass die pytorch collate fn arbeiten kann!
    # wofür brauch ich next iter überhaupt?

    print(next(iter(train_loader)))














''' Code Dumpster '''

# show random image if not specified
# if idx is None:
#     idx = (random.randrange(0, len(dataset),1), random.randrange(0, len(dataset[file_idx]['data']),1) )
# # if file_idx is None:
#     file_idx = random.randrange(0, len(dataset),1)
# if image_idx is None:
#     image_idx = random.randrange(0, len(dataset[file_idx]['data']),1)

# check index input
# assert (file_idx <= len(dataset)),"File idx must be smaller or equal to:\
#         {}".format(len(dataset))
# assert (image_idx <= len(dataset[file_idx]['data'])),\
#         "image idx must be smaller or equal to: {}"\
#         .format(len(dataset[file_idx]['data']))


# image_file = dataset[file_idx]
# print(image_file['targets'][image_idx][24])
# high_level_command = COMMAND_DICT[int(image_file['targets'][image_idx][24])]
# steering_angle = image_file['targets'][image_idx][0]
# img = image_file['data'][image_idx]

# print(sample['data'])



# print(train_set[0])
# show_image(train_set)

#
# print(type(train_set))
# n = random.randrange(0, len(train_set),1)
# # n = 2736
# print("Length of training set: {}".format(len(train_set)))
# foo = train_set[n]
# print(60* '-')
# print(60* '-')
# print("extracted example from file with index {}:".format(n))
# print(60* '-')
# print(foo['filename'])
# print(foo['data'])
# print(foo['targets'])
# print(60* '-')






#
#
# # example access
# #
# f = h5py.File('AgentHuman/SeqTrain/data_03663.h5', 'r')
#
# rgb_data = f['rgb']
# targets = f['targets']
#
# print("Keys in .h5 file: {}".format(list(f.keys())))
# print("RGB shape:     {}    \
# | Dtype: {}".format(rgb_data.shape, rgb_data.dtype))
# print("Targets shape: {}            \
# | Dtype: {}".format(targets.shape, targets.dtype))
#
#
# cv2.imshow('03663',rgb_data[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     plt.figure()
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
