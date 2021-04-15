import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import trange
from torch.autograd import grad



def load_binaryMNIST(batch_size):
    # load MNIST data set into data loader
    mnist_train = datasets.MNIST("./data", train=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False,  transform=transforms.ToTensor())

    idx_3, idx_7 = mnist_train.targets == 3, mnist_train.targets == 7
    idx_train = idx_3 | idx_7

    idx_3, idx_7 = mnist_test.targets == 3, mnist_test.targets == 7
    idx_test = idx_3 | idx_7

    mnist_train.targets = mnist_train.targets[idx_train]
    mnist_train.data = mnist_train.data[idx_train]
    mnist_test.targets = mnist_test.targets[idx_test]
    mnist_test.data = mnist_test.data[idx_test]

    # label 0: 3, label 1: 7 
    mnist_train.targets = ((mnist_train.targets - 3)/4).float()
    mnist_test.targets = ((mnist_test.targets - 3)/4).float()

    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=True)
    
    return train_loader, test_loader

avoid_zero_div = 1e-12
class pgd_rand(object):
    """ PGD attacks, with random initialization within the specified lp ball """
    def __init__(self, **kwargs):
        # define default attack parameters here:
        self.param = {'ord': np.inf,
                      'epsilon': 0.3,
                      'alpha': 0.01,
                      'num_iter': 40,
                      'restarts': 1,
                      'loss_fn': nn.BCEWithLogitsLoss()}
        # parse thru the dictionary and modify user-specific params
        self.parse_param(**kwargs) 
        
    def generate(self, model, x, y):
        epsilon = self.param['epsilon']
        alpha = self.param['alpha']
        num_iter = self.param['num_iter']
        restarts = self.param['restarts']
        loss_fn = self.param['loss_fn']
        p_norm = self.param['ord'] 
        
        # implementation begins:
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        max_delta = torch.zeros_like(x)
        _dim = x.shape[1] * x.shape[2] * x.shape[3]
        
        for i in range(restarts):
            if p_norm == np.inf:
                delta = torch.rand_like(x, requires_grad=True)
                delta.data = delta.data * 2. * epsilon - epsilon
                delta.data = (x.data + delta.data).clamp(min = 0, max = 1.0) - x.data
                for t in range(num_iter):
                    model.zero_grad()
                    loss = loss_fn(model(x + delta), y)
                    loss.backward()
                    # first we need to make sure delta is within the specified lp ball
                    delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(min = -epsilon, max = epsilon)
                    # then we need to make sure x+delta in the next iteration is within the [0,1] range
                    delta.data = (x.data + delta.data).clamp(min = 0, max = 1.) - x.data
                    delta.grad.zero_()
            
            # added the if condition to cut 1 additional unnecessary foward pass
            if restarts > 1:
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(x+delta),y)
                max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
                max_loss = torch.max(max_loss, all_loss)
            else:
                max_delta = delta.detach()
        return max_delta

    def parse_param(self, **kwargs):
        for key,value in kwargs.items():
            if key in self.param:
                self.param[key] = value

def one_epoch(model, device, loader, opt, train = True, adv = False):
    total_correct = 0
    if train:
        model.train()
    else:
        model.eval()

    with trange(len(loader)) as t:
        for X, y in loader:
            X, y = X.to(device), y.to(device).float().view(-1,1)

            opt.zero_grad()

            if adv:
                delta = pgd_rand().generate(model,X,y)
                yp = model(X+delta)
                loss = nn.BCEWithLogitsLoss()(yp, y)

            else:
                yp = model(X)
                loss = nn.BCEWithLogitsLoss()(yp, y)

            if train:
                loss.backward()
                opt.step()

            # ipdb.set_trace()
            batch_correct = ((yp>0) == y).sum().item()
            total_correct += batch_correct
            batch_acc = batch_correct / X.shape[0]
            # total_loss_adv += loss_adv.item() * X.shape[0]

            t.set_postfix(loss = loss.item(), 
                        batch_acc = '{0:.2f}%'.format(batch_acc*100), 
                        total_acc = '{0:.2f}%'.format(total_correct/loader.dataset.data.shape[0]*100))
            t.update()
            
    print("total accuracy:", total_correct/loader.dataset.data.shape[0]*100)
            
def mask_weight(model, mask_size, device):
    weight = list(model.parameters())[0].data.view(28,28).detach().cpu().numpy()
    # print(weight[10,10])
    weight_fft = np.fft.fftshift(np.fft.fft2(weight))

    mask = np.ones((28-mask_size*2, 28-mask_size*2))
    mask_padded = np.pad(mask, mask_size, 'constant', constant_values = 0)

    masked_weight_fft = np.multiply(weight_fft, mask_padded)

    masked_weight_ifft = np.fft.ifft2(np.fft.fftshift(masked_weight_fft))
    # masked_weight_ifft = np.fft.ifft2(np.fft.fftshift(weight_fft))

    list(model.parameters())[0].data = torch.tensor(np.real(masked_weight_ifft), dtype = torch.float, device = device).view(1,1,28,28)
    # print(list(model.parameters())[0].data[10,10])
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 28, 1, bias = False)
        torch.nn.init.xavier_uniform(self.conv1.weight)

    def forward(self, x):
        output = self.conv1(x).view(-1,1)
        return output