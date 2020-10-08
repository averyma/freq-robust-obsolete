import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def per_image_standardization(self, x):
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        mean = torch.mean(x, dim=(1,2,3), keepdim = True)
        stddev = torch.std(x, dim=(1,2,3), keepdim = True)
        adjusted_stddev = torch.max(stddev, (1./np.sqrt(_dim)) * torch.ones_like(stddev))
        return (x - mean) / adjusted_stddev

    def forward(self, x):
        x = self.per_image_standardization(x)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x