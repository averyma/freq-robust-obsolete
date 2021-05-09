import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ipdb

class two_layer_flatten(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(two_layer_flatten, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias = False)
        torch.nn.init.normal_(self.linear1.weight,mean=0.0, std=1.0)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
        torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)

    def forward(self, x):
        output = torch.sigmoid(self.linear1(x.t()))
        output = self.linear2(output)

        return output
    
class two_layer_conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(two_layer_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, input_dim, 1, bias = False)
#         torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
        torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)

    def forward(self, x):
        output = torch.sigmoid(self.conv1(x)[:,:,0,0])
        output = self.linear2(output)
        return output
    
# class NN_model_MNIST_relu(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NN_model_MNIST_relu, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, input_dim, 1, bias = False)
# #         torch.nn.init.xavier_uniform(self.conv1.weight)
#         torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
#         self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
#         torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)

#     def forward(self, x):
#         output = torch.nn.ReLU()(self.conv1(x)[:,:,0,0])
#         output = self.linear2(output)
#         return output
    
# class NN_model_MNIST_tanh(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NN_model_MNIST_tanh, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, input_dim, 1, bias = False)
# #         torch.nn.init.xavier_uniform(self.conv1.weight)
#         torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
#         self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
#         torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)

#     def forward(self, x):
#         output = torch.nn.Tanh()(self.conv1(x)[:,:,0,0])
#         output = self.linear2(output)
#         return output
    
# class NN_model_MNIST_sigmoid_uniform(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(NN_model_MNIST_sigmoid_uniform, self).__init__()
#         self.conv1 = nn.Conv2d(1, hidden_dim, input_dim, 1, bias = False)
#         torch.nn.init.xavier_uniform(self.conv1.weight)
# #         torch.nn.init.normal_(self.conv1.weight,mean=0.0, std=1.0)
#         self.linear2 = nn.Linear(hidden_dim, output_dim, bias = False)
# #         torch.nn.init.normal_(self.linear2.weight,mean=0.0, std=1.0)
#         torch.nn.init.xavier_uniform(self.linear2.weight)

#     def forward(self, x):
#         output = torch.sigmoid(self.conv1(x)[:,:,0,0])
#         output = self.linear2(output)
#         return output