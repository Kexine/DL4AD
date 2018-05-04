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
from torch.utils.data.sampler import  WeightedRandomSampler


import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda")



data_dir = 'TrainingSet'


### some statics

BATCH_SIZE = 256


# depth of first filter
depth_1 = 60

# depth of second filter
depth_2 = 100

# parameters for flattened array
num_param = 7*7*depth_2

# training epochs
num_train_epochs = 10

# learning rate
lr= 0.1

# logging lists
train_loss_epoch = []
list_val_loss = []
list_accuracy = []





### Neural Network Class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, depth_1, kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(depth_1, depth_2, kernel_size = 3, padding = 0, stride = 1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(num_param, 512)
        self.fc2 = nn.Linear(512, 43)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x),2))
        x = F.elu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  
        x = x.view(-1, num_param)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
    
    
### Global Methods 

# This function trains the neural network for one epoch
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Move the input and target data on the GPU
        data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        #loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {}| BatchIdx: {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader)*BATCH_SIZE,
                100. * batch_idx / len(train_loader), loss.item()))
        
        # loss per batch
        train_loss[epoch][batch_idx] = loss
        
        
    train_loss_epoch.append(np.mean(train_loss[epoch]))
       
    
# validation 
def validate():
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #val_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            
            # loss per validation batch
            val_loss += F.cross_entropy(output, target, size_average=True)
            
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
  
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    list_val_loss.append(val_loss)
    list_accuracy.append(100. * (correct / val_size))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, val_size,
        100. * (correct / val_size)))




### Transformation applied to Training Data

data_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1, 1, 1])
    ])


image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms)
dataset_size = len(image_dataset)


training_size = int(0.8*dataset_size)

val_size = dataset_size - training_size

train_weights = np.concatenate([np.ones(training_size), np.zeros(val_size)])

np.random.seed()
np.random.shuffle(train_weights)

val_weights = np.ones(len(train_weights)) - train_weights


# used WeightedRandomSampler to split Dataset, courtesy of Michael Floßmann
sample_splitter = {'Training': WeightedRandomSampler(train_weights, int(sum(train_weights)), replacement = False),
                  'Validation' : WeightedRandomSampler(val_weights, int(sum(val_weights)), replacement = False)}



print("Length of orig. training set: {}".format(dataset_size))
print("Splitted into:")
print("New training set of length {}".format(len(sample_splitter['Training'])))
print("New validation set of length {}".format(len(sample_splitter['Validation'])))




### Loader Objects
train_loader = DataLoader(image_dataset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True,
                             sampler = sample_splitter['Training'], drop_last = True)

val_loader = DataLoader(image_dataset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True,
                        sampler = sample_splitter['Validation'], drop_last = True)


### Some sanity checks
print("(SanityCheck) TrainLoader length: {}".format(len(train_loader)))
print("(SanityCheck) ValidationLoader length: {}".format(len(val_loader)))
print("val_loader.dataset length: {}".format(len(val_loader.dataset)))


model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

train_loss = np.zeros((num_train_epochs, len(train_loader)))

print("Shape of train_loss array: {}".format(train_loss.shape))

start = time.time()

for epoch in range(0, num_train_epochs):
    train(epoch)
    validate()


train_mean_loss = np.mean(train_loss, axis = 0)
print(train_mean_loss.shape)
                          
                          

end = time.time()
print("Traning time: {:.2f} min".format((end - start)/60))



# .csv savings
np.savetxt("trainLoss_{}.csv".format(lr), train_loss_epoch, delimiter=',')
np.savetxt("valiLoss_{}.csv".format(lr), list_val_loss, delimiter=',')


print("Train Loss list length: {}".format(len(train_loss)))                                            
print("Test Loss list length: {}".format(len(list_val_loss)))

# plotting
plt.figure()
plt.title("Training vs Validation Losses with LR {}".format(lr))
plt.plot(train_loss_epoch, label="Training Loss")
plt.plot(list_val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")


fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='black')
ax1.plot(train_loss_epoch, color='blue',label='Training Loss')
ax1.plot(list_val_loss, color='orange', label='Validation Loss')

ax1.tick_params(axis='y', labelcolor='black')

plt.legend()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


ax2.set_ylabel('Accuracy', color='black')  # we already handled the x-label with ax1
ax2.plot(list_accuracy, color='red', label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor='black')

fig.tight_layout()  # otherwise the right y-label is slightly clipped



plt.legend()

plt.show()
        
