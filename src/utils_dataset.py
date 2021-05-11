import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
    
def load_RealDataset(dataset, _batch_size = 128):

    if dataset in ["mnist","MNIST"]:
        train_loader, test_loader = load_MNIST(_batch_size)
    elif dataset in ["binarymnist","BinaryMnist","Binarymnist"]:
        train_loader, test_loader = load_binaryMNIST(_batch_size)
    elif dataset in ["fashionmnist","FashionMnist", "FashionMNIST"]:
        train_loader, test_loader = load_FashionMNIST(_batch_size)
    elif dataset in ["binarycifar10", "BinaryCifar10", "BinaryCIFAR10"]:
        train_loader, test_loader = load_binaryCIFAR(_batch_size)
    else:
        raise NotImplementedError("Dataset not included")
        
    return train_loader, test_loader
    
def load_binaryMNIST(batch_size):
    # load MNIST data set into data loader
    mnist_train = datasets.MNIST("./data", train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False,  transform=transforms.ToTensor())

    idx_3, idx_7 = mnist_train.targets == 3, mnist_train.targets == 7
    idx_train = idx_3 | idx_7

    idx_3, idx_7 = mnist_test.targets == 3, mnist_test.targets == 7
    idx_test = idx_3 | idx_7
    
    mnist_train.data = mnist_train.data[idx_train]
    mnist_test.data = mnist_test.data[idx_test]
        
    
    mnist_train.targets = mnist_train.targets[idx_train]
    mnist_test.targets = mnist_test.targets[idx_test]
    

    # label 0: 3, label 1: 7 
    mnist_train.targets = ((mnist_train.targets - 3)/4).float()
    mnist_test.targets = ((mnist_test.targets - 3)/4).float()

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder
    
    return train_loader, test_loader

def load_MNIST(batch_size):
    # load MNIST data set into data loader
    mnist_train = datasets.MNIST("./data", train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False,  transform=transforms.ToTensor())

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=True)
    
    return train_loader, test_loader

def load_FashionMNIST(batch_size):
    # load MNIST data set into data loader
    fmnist_train = datasets.FashionMNIST("./data", train=True, download = True, transform=transforms.ToTensor())
    fmnist_test = datasets.FashionMNIST("./data", train=False,  download = True, transform=transforms.ToTensor())

    train_loader = DataLoader(fmnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(fmnist_test, batch_size = batch_size, shuffle=True)
    
    return train_loader, test_loader

def load_grayCIFAR(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    data_train = datasets.CIFAR10("./data", train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10("./data", train=False, download = True, transform=transform_test)
    
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder

    return train_loader, test_loader

def load_binaryCIFAR(batch_size, target1=1, target2=5):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])

    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])
    
    # load CIFAR data set into data loader
    data_train = datasets.CIFAR10("./data", train=True, download = True, transform=transform_train)
    data_test = datasets.CIFAR10("./data", train=False, download = True, transform=transform_test)
    
     
    idx_3, idx_7 = torch.tensor(data_train.targets) == target1, torch.tensor(data_train.targets) == target2
    idx_train = (idx_3 | idx_7).tolist()

    idx_3, idx_7 = torch.tensor(data_test.targets) == target1, torch.tensor(data_test.targets) == target2
    idx_test = (idx_3 | idx_7).tolist()
    
    data_train.data = data_train.data[idx_train,:,:,:]
    data_test.data = data_test.data[idx_test,:,:,:]
        
    data_train.targets = torch.tensor(data_train.targets)[idx_train].tolist()
    data_test.targets = torch.tensor(data_test.targets)[idx_test].tolist()

    # label 0: 3, label 1: 7
    idx_train = (torch.tensor(data_train.targets) == target1)
    data_train.targets = idx_train.float().tolist()
    idx_test = (torch.tensor(data_test.targets) == target1)
    data_test.targets = idx_test.float().tolist()
    
    train_loader = DataLoader(data_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size = batch_size, shuffle=True)
    
    class_holder = ['0 - zero',
                    '1 - one']
    train_loader.dataset.classes = class_holder
    test_loader.dataset.classes = class_holder
    
    return train_loader, test_loader