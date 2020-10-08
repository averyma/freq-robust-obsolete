import torch
import torch.nn as nn
import numpy as np

avoid_zero_div = 1e-12

class pgd_rand(object):
    """ PGD attacks, with random initialization within the specified lp ball """
    def __init__(self, **kwargs):
        # define default attack parameters here:
        self.param = {'ord': np.inf,
                      'epsilon': 8./255.,
                      'alpha': 2./255.,
                      'num_iter': 20,
                      'restarts': 1,
                      'loss_fn': nn.CrossEntropyLoss()}
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

            elif p_norm == 2:
                delta = torch.rand_like(x, requires_grad = True)
                delta.data = delta.data * 2.0 - 1.0
                delta_norm = torch.norm(delta.detach(), p = 2 , dim = (1,2,3), keepdim = True).clamp(min = avoid_zero_div)
                delta.data = epsilon * delta.data/delta_norm
                delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data
                for t in range(num_iter):
                    model.zero_grad()
                    loss = loss_fn(model(x + delta), y)
                    loss.backward()

                    # computing norm of loss gradient wrt input
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    grad_norm = torch.norm(delta.grad.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # one step in the direction of normalized gradient (stepsize = alpha)
                    delta.data = delta + alpha*delta.grad.detach()/grad_norm
                    # computing the norm of the new delta term
                    # To avoid zero division, cleverhans uses 1e-12, advertorch uses 1e-6
                    delta_norm = torch.norm(delta.detach().view(-1, _dim), p = 2 , dim = 1).view(x.shape[0],1,1,1).clamp(min = avoid_zero_div)
                    # here the clip factor is used to **clip** to within the norm ball
                    # not to **normalize** onto the surface of the ball
                    factor = torch.min(epsilon/delta_norm, torch.tensor(1., device = x.device ))

                    delta.data = delta.data * factor
                    delta.data = (x.data + delta.data).clamp(min = 0., max = 1.) - x.data

                    delta.grad.zero_()
            else: 
                error = "Only ord = inf and ord = 2 have been implemented"
                raise NotImplementedError(error)
            
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
