import random
import os
import operator as op
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models import c11, ResNet18, ResNet8
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import grad
    
def seed_everything(manual_seed):
    # set benchmark to False for EXACT reproducibility
    # when benchmark is true, cudnn will run some tests at
    # the beginning which determine which cudnn kernels are
    # optimal for opertions
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(argu, device):
    if argu.arch == "c11":
        model = c11()
    elif argu.arch == "resnet18":
        model = ResNet18()
    elif argu.arch == "resnet8":
        model = ResNet8()
    
    if argu.pretrain:
        
        model.load_state_dict(torch.load(argu.pretrain, map_location=device))
        model.to(device)
        print("\n ***  pretrain model loaded: "+ argu.pretrain + " *** \n")

    return model.to(device)

def get_optim(model, argu):
    """
    recommended setup:
    SGD_step: initial lr:0.1, momentum: 0.9, weight_decay: 0.0002, miliestones: [100, 150]
    Adam_step: initial lr:0.1, milestones: [80,120,160,180]
    others: constant lr at 0.001 should be sufficient
    """
    
    if "sgd" in argu.optim:
        opt = optim.SGD(model.parameters(), lr = argu.lr, momentum = argu.momentum, weight_decay = argu.weight_decay)
    elif "adam" in argu.optim:
        opt = optim.Adam(model.parameters(), lr = argu.lr)
   
    # check if milestone is an empty array
    if argu.lr_update == "multistep":
        _milestones = [argu.epoch/ 2, argu.epoch * 3 / 4]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=_milestones, gamma=0.1)
    elif argu.lr_update == "fixed":
        lr_scheduler = False

    return opt, lr_scheduler

class DictWrapper(object):
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

def ep2itr(epoch, loader):
    data_len = loader.dataset.data.shape[0]
    batch_size = loader.batch_size
    iteration = epoch * np.ceil(data_len/batch_size)
    return iteration
