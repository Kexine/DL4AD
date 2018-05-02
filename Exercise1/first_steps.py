#!/usr/bin/env python

import torch
import numpy as np

from torch.autograd.variable import Variable
x = Variable(torch.ones((2,2)), requires_grad = True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)
