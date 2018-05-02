#!/usr/bin/env python
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode

### Global Functions ###########################################################


def show_landmarks(image, landmarks):
    # extract coordinates landmakrs in a more reader friendly version
    x1, y1, x2, y2 = landmarks[2:6]
    plt.imshow(image)
    # plot edges of landmarks
    plt.plot([x1,x2], [y1,y1], c='r', marker='o')
    plt.plot([x1,x2], [y2,y2], c='r', marker='o')
    plt.plot([x1,x1], [y1,y2], c='r', marker='o')
    plt.plot([x2,x2], [y1,y2], c='r', marker='o')
    plt.pause(0.001)  # pause a bit so that plots are updated

class StreetSignLandmarkDataset(Dataset):
    """Street Sign Landmark Dataset
        Args:
            csv_file (string): Path to the csv file with annotations
            root_dir (string): Directory with the images
            transform (Callables, optional): optional transform to be applied
                on a sample.
        """
    def __init__(self, csv_file, root_dir, transform = None):
        self.landmarks_frame = pd.read_csv(csv_file, sep=';')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        self.img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,0])
        image = io.imread(self.img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('int').reshape(-1,7)
        landmarks = landmarks.squeeze()
        sample = {'image' : image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size"""
    """
    Args:
        output_size (tuples or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * h / w
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # print(landmarks)
        # print(new_w, w, new_h, h)
        # landmarks[:] = landmarks[:] * [new_w / w, new_h / h]

        landmarks[0] = landmarks[0] * (new_w / w)
        landmarks[1] = landmarks[1] * (new_h / h)
        # print(landmarks[2])
        landmarks[2] = landmarks[2] * (new_w / w)
        landmarks[3] = landmarks[3] * (new_h / h)

        landmarks[4] = landmarks[4] * (new_w / w)
        landmarks[5] = landmarks[5] * (new_h / h)


        # print(landmarks)

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        print(h, new_h, top)
        print(w, new_w, left)
        print(landmarks)
        # # landmarks = landmarks - [left, top]
        landmarks[0] -= new_w
        landmarks[1] -= new_h
        # print(landmarks[2])
        landmarks[2] = (landmarks[2]  - (left + new_w)) * (new_w / w)
        landmarks[3] = (landmarks[3]  - (top + new_h)) * (new_h / h)
        landmarks[4] = (landmarks[4]  - (left + new_w)) * (new_w / w)
        landmarks[5] = (landmarks[5]  - (top + new_h)) * (new_h / h)
        #
        # landmarks[3] = landmarks[3] * (new_h / h) - (top + new_h)
        #
        # landmarks[4] = landmarks[4] * (new_w / w) - (left + new_w)
        # landmarks[5] = landmarks[5] * (new_h / h) - (top + new_h)

        print(landmarks)
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        print(image.shape)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

################################################################################


# device = torch.device("cuda")
#
# landmarks_frame = pd.read_csv('TrainingSet/00000/GT-00000.csv',sep = ';')
# n = 2
#
#
# img_name = landmarks_frame.iloc[n, 0]
# landmarks = landmarks_frame.iloc[n,1:].as_matrix()
# landmarks = landmarks.astype('int').reshape(-1,7)
# landmarks = landmarks.squeeze()
# print("Image Name: {}".format(img_name))
# print("Landmaks Shape: {}".format(landmarks.shape))
# print("First 4 Landmarks: {}".format(landmarks[:4]))
# print("All Landmarks: {}".format(landmarks[:]))
#
#
# show_landmarks(io.imread(os.path.join('TrainingSet/00000', img_name)), landmarks)
#

sign_dataset = StreetSignLandmarkDataset(
                    csv_file = 'TrainingSet/00000/GT-00000.csv',
                    root_dir = 'TrainingSet/00000')


transformed_dataset = StreetSignLandmarkDataset(csv_file = 'TrainingSet/00000/GT-00000.csv',
root_dir = 'TrainingSet/00000', transform=transforms.Compose([Rescale((32,32)),ToTensor()]))

dataloader = DataLoader(transformed_dataset, batch_size=1, shuffle=True, num_workers=4)


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    print(landmarks_batch)
    for i in range(batch_size):
        # plt.scatter(landmarks_batch[i, 0].numpy() + i * im_size,
        #             landmarks_batch[i, 1].numpy(),
        #             s=10, marker='.', c='r')
        x1, y1, x2, y2 = landmarks_batch[i][2:6]
        # print(i,x1,y1,x2,y2)

        plt.plot([x1,x2], [y1,y1], c='r', marker='o')
        plt.plot([x1,x2], [y2,y2], c='r', marker='o')
        plt.plot([x1,x1], [y1,y2], c='r', marker='o')
        plt.plot([x2,x2], [y1,y2], c='r', marker='o')
        # plt.pause(0.001)  # pause a bit so that plots are updated

        plt.title('Batch from dataloader')

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
