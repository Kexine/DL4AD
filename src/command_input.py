#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
end to end driving

exercise 3: command input architecture 

by: Michael Floßmann, Kshitij Sirohi, Hendrik Vloet

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
device = torch.device("cuda")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #8 layers if con layers for image module
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2) #(output = 200x88)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)#(output = 200x88)   
        self.conv3= nn.Conv2d(32, 64, kernel_size=3, padding=2) #(output = 202x90)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#(output = 202x90)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=2)#(output = 204x92)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)#(output = 204x92)   
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)#(output = 204x92)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)#(output = 204x92)
        
        
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
        self.fc1 = nn.Linear(204*92*256, 512) #(please reconfirm with team)
        self.fc2 = nn.Linear(512, 512)
        
        #3 fc layers for control and measurement modules
        self.fc3= nn.Linear(1,128)
        self.fc4= nn.Linear(128,128)
        
        #4 fc layers for concatenated module
        self.fc5= nn.Linear(768,512)
        self.fc6= nn.Linear(512,256)
        self.fc7= nn.Linear(256,256)
        
        #5 for action output
        self.fc8= nn.Linear(256,2)
        
        
    def forward(self, x,m,c):
        #######conv layers##############
        x= self.conv1(x)
        x= self.conv1_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        
        x= self.conv2(x)
        x= self.conv2_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv3(x)
        x= self.conv3_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv4(x)
        x= self.conv4_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv5(x)
        x= self.conv5_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv6(x)
        x= self.conv6_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv7(x)
        x= self.conv7_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        x= self.conv8(x)
        x= self.conv8_bn(x)
        x= self.conv_drop(x)
        x = relu(x)
        
        ###################################
        
        x = x.view(-1, 204*92*256)      ### do change this
        
        
        #########fully connected layers####
        x = self.fc1(x)
        x = self.fc_drop(x)
        x = relu(x)
        
        x = self.fc2(x)
        x= self.fc_drop(x)
        x = relu(x)
        
        
        ####################################
        
        ####for  measurement(speed)#########
        m = self.fc3(m)
        x= self.fc_drop(x)
        x = relu(x)
        
        m = self.fc4(m)
        m= self.fc_drop(m)
        m = relu(m)
        ####################################
        
        #########for control################
        c = self.fc3(c)
        c = self.fc4(c)
        
        ###concatenating previous layers####
        j = torch.cat((x,m,c), 1)
        j = self.fc5(j)
        j= self.fc_drop(j)
        j = relu(j)
        
        ####################################
        j = self.fc6(j)
        j= self.fc_drop(j)
        j = relu(j)
        
        j = torch.cat((x,m,c), 1)
        j = self.fc7(j)
        j= self.fc_drop(j)
        j = relu(j)
        
        
        #### output action##########
        action = self.fc8(j)
        
        return action


model = Net().to(device)

relu = F.relu

optimizer = optim.Adam(model.parameters(), lr=0.0002)

criterion = nn.CrossEntropyLoss()

lossx = []

