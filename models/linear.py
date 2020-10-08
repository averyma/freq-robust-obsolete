"""
AM: Aug28, 2019 rename model l1, l2 and l3 as linear1, linear2 and linear 3
    edited main.py to reflect change, for binary_mnist_l2*, i changed their 
    name to binary_mnist_linear2 in both the actual weight file and the record
    in the log.txt file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class linear1(nn.Module):
    def __init__(self):
        super(linear1, self).__init__()
        self.l = nn.Linear(784, 10, bias = True)
        
        # self.l.weight.data = torch.zeros_like(self.l.weight.data)
        # self.l.weight.data = torch.sign(torch.rand_like(self.l.weight.data))*0.1
        # self.l.weight.data[0]= 0.1
        # self.l.weight.data[1]= -0.1
        # self.l.bias.data = torch.zeros_like(self.l.bias.data)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.l(x)

class linear2(nn.Module):
    def __init__(self):
        super(linear2, self).__init__()
        self.l = nn.Linear(784, 1, bias = True)
        self.sig = nn.Sigmoid()
        
        # self.l.weight.data = torch.zeros_like(self.l.weight.data)
        # self.l.weight.data = torch.sign(torch.rand_like(self.l.weight.data))*0.1
        # self.l.weight.data[0]= 0.1
        # self.l.weight.data[1]= -0.1
        # self.l.bias.data = torch.zeros_like(self.l.bias.data)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.sig(self.l(x))

class linear3(nn.Module):
    def __init__(self):
        super(linear3, self).__init__()
        self.l = nn.Linear(784, 10, bias = True)
        self.sig = nn.Sigmoid() 
        # self.l.weight.data = torch.zeros_like(self.l.weight.data)
        # self.l.weight.data = torch.sign(torch.rand_like(self.l.weight.data))*0.1
        # self.l.weight.data[0]= 0.1
        # self.l.weight.data[1]= -0.1
        # self.l.bias.data = torch.zeros_like(self.l.bias.data)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.sig(self.l(x))

class linear5(nn.Module):
    """
    Implementation of the "2-hidden-layer ReLU network with 1000 hidden units" used in the 
    adversarial sphere paper: https://arxiv.org/abs/1801.02774
    """
    def __init__(self):
        super(linear5, self).__init__()
        self.layer1 = nn.Linear(500, 1000, bias = True)
        self.layer2 = nn.Linear(1000, 1000, bias = True)
        self.readout = nn.Linear(1000, 2, bias = True)
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(1000)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.readout(x)
    
        return x
    
class quad1(nn.Module):
    """
    Implementation of the "quadratic network" used in the 
    adversarial sphere paper: https://arxiv.org/abs/1801.02774
    """
    def __init__(self):
        super(quad1, self).__init__()
        self.layer1 = nn.Linear(500, 1000, bias = False)
        self.readout = nn.Linear(1, 1, bias = True)

    def act(self, x):
        return x**2
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = x.sum(dim = 1, keepdim = True)
        x = self.readout(x)
    
        return x.squeeze()
