#!/usr/bin/env python3

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda")


### Transformation applied to Training Data

data_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ])

data_dir = 'TrainingSet'

image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms)

train_loader =  DataLoader(image_dataset, batch_size= 1,
                                              shuffle=True, num_workers=4, drop_last = True,
                                              pin_memory = True)

class_names = image_dataset.classes

print("Length of dataset: {}".format(len(train_loader)))
print("Amount of classes: {}".format(len(class_names)))


train_data = []
val_data = []

foo = np.array(np.arange(len(train_loader)))
np.random.shuffle(foo)


for (idx, val) in enumerate(foo):
    if (idx < int(0.8*len(image_dataset))):
        train_data.append(image_dataset[val])
    else:
        val_data.append(image_dataset[val])

# plt.figure()
# plt.imshow(train_data[2][0].numpy().transpose((1,2,0)))
# plt.show()

for batch_idx, (data, target) in enumerate(train_data):
    print(data, target)
    plt.figure()
    plt.imshow(data.numpy().transpose((1,2,0)))
    plt.show()
    exit()

print(len(train_data))
print(len(val_data))



exit()



print("Length of the whole Training Set: {}".format(dataset_length))
print("Amount of Batches when using a 80/20 Train/Val Ratio:{}".format(len(dataloaders['TrainingSet'])))
print("Elements in one Batch: {}".format(len(inputs)))


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
#out = torchvision.utils.make_grid(inputs[0])

#imshow(out, title=[class_names[x] for x in classes])
#plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size = 3, padding = 2, stride = 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 3, padding = 0, stride = 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 43)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print("x.shape: {}".format(x.shape))
        x = x.view(-1, 1280)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# This function trains the neural network for one epoch
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloaders['TrainingSet']):
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        loss = F.nll_loss(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloaders.dataset),
                100. * batch_idx / len(dataloaders), loss.item()))


num_train_epochs = 10
for epoch in range(1, num_train_epochs + 1):
    train(epoch)
