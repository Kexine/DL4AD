#!/usr/bin/env python3

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


### Transformation applied to Training Data

data_transforms = {
    'TrainingSet': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ])
}
        
data_dir = 'Images'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['TrainingSet']}

dataset_length = len(image_datasets['TrainingSet'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= int(dataset_length * 0.2) , 
                                              shuffle=True, num_workers=4, drop_last = True,
                                              pin_memory = True)
                                                    for x in ['TrainingSet']}


print(dataset_length)
print(len(dataloaders['TrainingSet']))

# Get a batch of training data
inputs, classes = next(iter(dataloaders['TrainingSet']))

class_names = image_datasets['TrainingSet'].classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    


# Make a grid from batch
out = torchvision.utils.make_grid(inputs[0])

imshow(out, title=[class_names[x] for x in classes])
plt.show()
