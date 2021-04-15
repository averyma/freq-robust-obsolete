import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from src.utils_freq import dct, idct

from collections import defaultdict
from tqdm import trange
import ipdb

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



def loader_LR(case, d, batchsize, mu, std, lambbda, iteration):
    
    total_size = batchsize*iteration
    
    x, y = data_init_LR(case, d, total_size, mu, std, lambbda)
    dataset = TensorDataset(x.t(), y.t())
    loader = DataLoader(dataset, batch_size = batchsize, pin_memory = True)
    
    return loader



def data_init_LR(case = 1, d = 10, batchsize = 128, mu = 1, std = 0.5, lambbda = 1):
    
    w_star = torch.zeros(d, 1)
    w_star[0] = 1
    
    if case == 1:
        x_tilde = torch.zeros(d, batchsize)
        x_tilde[0,:] = torch.normal(mean = mu, std = std, size = (1,batchsize))
    elif case == 2:
        x_tilde = torch.normal(mean = mu, std = std, size = (d, batchsize))
    elif case == 3:
#         alpha = torch.normal(mean = mu, std = std, size = (1, batchsize)).repeat(d,1)
        alpha = torch.normal(mean = mu, std = std, size = (d, batchsize))
        decay = torch.exp(-lambbda*(torch.range(1, d)-1).view(d,1).repeat(1, batchsize))
        x_tilde = alpha * decay
    
    x = idct(x_tilde)
    y = torch.mm(w_star.t(), x_tilde)
    return x, y

def grad_estimation(w_tilde, lr, case, d, mu, std, lambbda):
    
    grad_est = torch.zeros(d,1, device= w_tilde.device)
    
    if case == 1:
        grad_est[0] = lr*(w_tilde[0]-1)*(mu**2 + std**2)
        
    elif case == 2:
        grad_est[0] = lr*(w_tilde[1:].sum()*mu**2 + (w_tilde[0]-1)*(mu**2 + std**2))
        
        for i in range(1, d):
            grad_est[i] = lr*(w_tilde.sum()*mu**2 - w_tilde[i]*mu**2 + w_tilde[i]*(mu**2 + std**2) - mu**2)
            
    elif case == 3:
        exp_i = torch.exp(-lambbda*(torch.range(1, d, device = w_tilde.device)-1))
        grad_est[0] = lr*( (mu**2 * w_tilde * exp_i)[1:].sum() + (w_tilde[0]-1)*(mu**2 + std**2))
        
        for i in range(1, d):
            exp_k = torch.exp(-lambbda*torch.tensor(i))
            grad_est[i] = lr*(((w_tilde*exp_i*exp_k).sum() - w_tilde[i]*exp_k**2)*mu**2 + w_tilde[i]*exp_k**2*(mu**2 + std**2) - mu**2*exp_k)

    return grad_est

def train_LR(args, model, opt, device):
    
    iteration = args["itr"]
    _d = args["d"]
    _case = args["case"]
    _batchsize = args["bsize"]
    _mu = args["mu"]
    _std = args["std"]
    _lambbda = args["lambbda"]
    _lr = args["lr"]
    
    log_dict = defaultdict(lambda: list())
    
    w = torch.zeros(_d, iteration, device = device)
    w_tilde =  torch.zeros(_d, iteration, device = device)
    dw_tilde =  torch.zeros(2, _d, iteration, device = device)
    loss_logger = torch.zeros(1, iteration, device = device)
    loss_adv_logger = torch.zeros(3, iteration, device = device) # 3 types of attacks
    
    
    train_loader = loader_LR(_case, _d, _batchsize, _mu, _std, _lambbda, iteration)
    
    i = 0
#     with trange(iteration) as t:
    for x, y in train_loader:

        x, y = x.t().to(device), y.t().to(device)

        w_tilde_prev = dct(list(model.parameters())[0][0].view(_d,1)).squeeze().detach()

        opt.zero_grad()

        y_hat = model(x)

        loss = ((1/2) * (y_hat.t() - y) ** 2).mean()

        loss_logger[:,i] = loss.item()
    
        loss_adv = loss_under_attack(args, model, device)

        loss.backward()
        opt.step()

        w[:,i] = list(model.parameters())[0][0].detach()
        w_tilde[:,i] = dct(w[:,i].view(_d,1)).squeeze().detach()
        dw_tilde[0,:,i] = -(w_tilde[:,i] - w_tilde_prev).detach()
        dw_tilde[1,:,i] = grad_estimation(w_tilde_prev, 
                                          _lr, 
                                          _case, 
                                          _d, 
                                          _mu, 
                                          _std, 
                                          _lambbda).squeeze().detach()
        loss_adv_logger[:,i] = loss_adv
        
        i += 1
            
    log_dict["w"] = w
    log_dict["w_tilde"] = w_tilde
    log_dict["dw_tilde"] = dw_tilde
    log_dict["loss"] = loss_logger
    log_dict["loss_adv"] = loss_adv_logger

    return log_dict

def loss_under_attack(args, model, device):
    
    loss_adv = torch.zeros(3, device = device) # 3 types of attacks
    _d = args["d"]
    _case = args["case"]
    _batchsize = args["bsize"]
    _mu = args["mu"]
    _std = args["std"]
    _lambbda = args["lambbda"]
    _lr = args["lr"]
    _eps = args["eps"]
    
    x, y = data_init_LR(_case, _d, _batchsize, _mu, _std, _lambbda) 
    x, y = x.to(device), y.to(device)
    
    
    y_hat = model(x)
    r = (y_hat.t() - y)
    
    w = list(model.parameters())[0][0].detach()
    w_tilde = dct(w.view(_d,1)).detach()
    

#     #attack 1: all k's
#     delta_x_1_tilde = _eps * torch.sign(r) * w_tilde / torch.norm(w_tilde).detach()
#     delta_x_1 = idct(delta_x_1_tilde)
#     y_adv_1 = model(x+delta_x_1)
#     loss_1 = ((1/2) * (y_adv_1.t() - y) ** 2).mean().item()
    
#     #attack 2: highest k
#     attack_k = torch.ones(_d, device = device)
#     attack_k[0:-1] = 0
#     V = torch.diag(attack_k)
#     V_w_tilde = torch.mm(V, w_tilde)
#     delta_x_2_tilde = _eps * torch.sign(r) * V_w_tilde / torch.norm(V_w_tilde).detach()
#     delta_x_2 = idct(delta_x_2_tilde)
#     y_adv_2 = model(x+delta_x_2)
#     loss_2 = ((1/2) * (y_adv_2.t() - y) ** 2).mean().item()
    
    #attack 3: k != 0
    attack_k = torch.ones(_d, device = device)
    attack_k[0] = 0
    V = torch.diag(attack_k)
    V_w_tilde = torch.mm(V, w_tilde)
#     ipdb.set_trace()
#     if _case == 1:
#         print(w_tilde)
    delta_x_3_tilde = _eps * torch.sign(r) * V_w_tilde / torch.norm(V_w_tilde).detach()
    delta_x_3 = idct(delta_x_3_tilde)
    y_adv_3 = model(x+delta_x_3)
    loss_3 = ((1/2) * (y_adv_3.t() - y) ** 2).mean().item()
    
    
    loss_adv[0], loss_adv[1], loss_adv[2] = 0, 0, loss_3
    
    return loss_adv
    
def plot_loss_LR(log, threshold = 1e-3, plot_itr = 1000):
    
    THRESHOLD = threshold
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,1)
    
    loss_var, loss_mean  = torch.var_mean(log, dim = 1) 
    fill_up = loss_mean + loss_var
    fill_low = loss_mean - loss_var

    xrange = np.arange(log.shape[0])  
    
    loss_below_threshold = loss_mean < THRESHOLD
    
    
    for i in range(log.shape[2]):
        fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1), linewidth=3.0, marker = "")
        fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
        try:
            fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
        except ValueError as e:
            print("Above loss threshold! (loss plot)")
            
    fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[0,0]).set_ylabel("loss")
    fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
    fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
def plot_risk_LR(args, w_tilde_log, loss_log, threshold = 1e-3, plot_itr = 1000):
    
    THRESHOLD = threshold
    
    _std = args["std"]
    _lr = args["lr"]
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,1)
    
    loss_var, loss_mean  = torch.var_mean(loss_log, dim = 1) 
    fill_up = loss_mean + loss_var
    fill_low = loss_mean - loss_var

    xrange = np.arange(loss_log.shape[0])  
    
    loss_below_threshold = loss_mean < THRESHOLD
    
    
    for i in range(loss_log.shape[2]):
        fig.add_subplot(gs[0,0]).plot(loss_mean[:, i], color = "C"+str(i), label = "case "+str(i+1), linewidth=3.0, marker = "")
        fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
        
        if i == 0: # case 1
            e_0 = w_tilde_log[0,0,:,0] - 1 # only supports numb_runs = 1, so error will occur if we do average over multiple runs
            risk = 0.5 * e_0**2 * _std**2 * torch.tensor(1 - 2*_lr*_std**2 + 3*_lr**2*_std**4)**torch.tensor(xrange)
            fig.add_subplot(gs[0,0]).plot(risk, color = "C"+str(i+3), label = "case "+str(i+1)+" risk", linewidth=3.0, marker = "")
        elif i == 1: # case 2
            pass
        elif i ==2: # case 3
            pass
        
        try:
            fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
        except ValueError as e:
            print("Above loss threshold! (loss plot)")
            
    fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])
            
        
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[0,0]).set_ylabel("loss")
    fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
    fig.add_subplot(gs[0,0]).set_title("loss",fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    
def plot_w_tilde_LR(log, threshold = 1e-3):
    
    THRESHOLD = threshold
    
    fig = plt.figure(figsize = [15,15])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(10,1)
    
    w_tilde_log = log.clone().detach()
    w_tilde_log[0,:,:,:] = w_tilde_log[0,:,:,:]-1
    
    w_tilde_diff_var, w_tilde_diff_mean  = torch.var_mean(w_tilde_log, dim = 2, unbiased = True)

    xrange = np.arange(log.shape[1])
    
    w_tilde_diff_below_threshold = w_tilde_diff_mean.abs() < THRESHOLD
    
    for i in range(log.shape[3]):
        for j in range(10):
            fig.add_subplot(gs[j,0]).plot(w_tilde_diff_mean[j,:,i], color = "C"+str(i), label = "case " + str(i+1), linewidth=3.0, marker = "")
#             fig.add_subplot(gs[j,0]).fill_between(xrange, fill_up[j,:,i-1], fill_low[j,:,i-1], color = "C"+str(i), alpha=0.3)
            try:
                fig.add_subplot(gs[j,0]).axvline(x=w_tilde_diff_below_threshold[j,:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
            except ValueError as e:
                print("Above loss threshold! (w_tilde plot) ", i, j)
    
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10}) 
    fig.add_subplot(gs[0,0]).set_title("$\~e(k)$",fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[5,0]).set_ylabel("Frequency")
    fig.add_subplot(gs[9,0]).set_xlabel("Training iteration")
    
    fig.tight_layout()


def plot_dw_tilde_LR(log):
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
#     gs = fig.add_gridspec(10,1)
    
#     dw_tilde_log = log.clone().detach()
    
#     for j in range(10):
#         fig.add_subplot(gs[j,0]).plot(dw_tilde_log[0,j,:].cpu().numpy(), color = "C1", label = "actual", linewidth=3.0, marker = "")
#         fig.add_subplot(gs[j,0]).plot(dw_tilde_log[1,j,:].cpu().numpy(), color = "C2", label = "estimated", linewidth=3.0, marker = "")
        
    gs = fig.add_gridspec(1,1)
    
    dw_tilde_log = log.clone().detach()
    
    for j in range(10):
        fig.add_subplot(gs[0,0]).plot(dw_tilde_log[0,j,:].cpu().numpy(), color = "C"+str(j), label = "actual"+str(j), linewidth=3.0, marker = "",alpha=0.2)
        fig.add_subplot(gs[0,0]).plot(dw_tilde_log[1,j,:].cpu().numpy(), color = "C"+str(j), label = "estimated"+str(j), linewidth=2.0, marker = "")
        
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})        
#     fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
    fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    
def plot_loss_adv_LR(log, threshold = 1e-3, plot_itr = 1000):
    
    THRESHOLD = threshold
    
    
    fig = plt.figure(figsize = [15,7])
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1,1)
    
    
    
    loss_var, loss_mean  = torch.var_mean(log, dim = 2)
#     print(loss_mean.shape)
    fill_up = loss_mean + loss_var
    fill_low = loss_mean - loss_var

    xrange = np.arange(log.shape[0])
    

    
    loss_below_threshold = loss_mean < THRESHOLD
    
#     print(loss_below_threshold.shape)
    
    #loss_mean: [iteration, attacks, case]
#     for _case in range(loss_mean.shape[2]):
#         for _attack in range(loss_mean.shape[1]):
#             fig_label = " attack "+str(_attack+1)
#             fig.add_subplot(gs[_case,0]).plot(loss_mean[:, _attack, _case], color = "C"+str(_attack), label = fig_label, linewidth=3.0, marker = "")
        
#         fig.add_subplot(gs[_case,0]).set_title("case "+str(_case+1))
#         fig.add_subplot(gs[_case,0]).legend(prop={"size": 10})
        
        
#     for _attack in range(loss_mean.shape[1]):
#         for _case in range(loss_mean.shape[2]):
#             fig_label = " case "+str(_case+1)
#             fig.add_subplot(gs[_attack+3,0]).plot(loss_mean[:, _attack, _case], color = "C"+str(_case), label = fig_label, linewidth=3.0, marker = "")
        
#         fig.add_subplot(gs[_attack+3,0]).set_title("attack "+str(_attack+1))
#         fig.add_subplot(gs[_attack+3,0]).legend(prop={"size": 10})
        
    for _case in range(loss_mean.shape[2]):
        fig_label = " case "+str(_case+1)
        fig.add_subplot(gs[0,0]).plot(loss_mean[:, 2, _case], color = "C"+str(_case), label = fig_label, linewidth=3.0, marker = "")
        try:
            fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,2,_case].tolist().index(1), color = "C"+str(_case), linestyle = '--')
        except ValueError as e:
            print("Above loss threshold")
            
    fig.add_subplot(gs[0,0]).set_xlim([0, plot_itr])

#     fig.add_subplot(gs[0,0]).set_title("attack "+str(_case+1))
    title_text = "loss under attack with "+ r"$ \Delta x = iDCT\{ \epsilon sign(r) \frac{V\~w}{||V\~w||}\}$ where $V =diag\{0,1,...,1\}$"
    fig.add_subplot(gs[0,0]).set_title(title_text,fontsize = 20)
    fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
        
        
        
#         fig.add_subplot(gs[0,0]).fill_between(xrange, fill_up[:, i], fill_low[:, i], color = "C"+str(i), alpha=0.3)
#         try:
#             fig.add_subplot(gs[0,0]).axvline(x=loss_below_threshold[:,i].tolist().index(1), color = "C"+str(i), linestyle = '--')
#         except ValueError as e:
#             print("Above loss threshold")
            
        
#     fig.add_subplot(gs[0,0]).legend(prop={"size": 10})
    fig.add_subplot(gs[0,0]).set_ylabel("loss")
    fig.add_subplot(gs[0,0]).set_xlabel("Training iteration")
    fig.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    
    
    
    
    
    
### OBSOLETE ###   
# def train_LR_optim_lr(args, model, device):
    
#     iteration = args["itr"]
#     _d = args["d"]
#     _case = args["case"]
#     _batchsize = args["bsize"]
#     _mu = args["mu"]
#     _std = args["std"]
#     _lambbda = args["lambbda"]
#     _lr = args["lr"]
    
#     log_dict = defaultdict(lambda: list())
    
#     w = torch.zeros(_d, iteration, device = device)
#     w_tilde =  torch.zeros(_d, iteration, device = device)
#     dw_tilde =  torch.zeros(2, _d, iteration, device = device)
#     loss_log = torch.zeros(1, iteration, device = device)
    
#     optim_init = False
# #     opt = optim.SGD(model.parameters(), lr = 1e-2)
    
#     with trange(iteration) as t:
#         for i in range(iteration):

#             x, y = data_init_LR( case = _case, 
#                                  d = _d, 
#                                  batchsize = _batchsize, 
#                                  mu = _mu, 
#                                  std = _std, 
#                                  lambbda = _lambbda)

#             x, y = x.to(device), y.to(device)

#             w_tilde_prev = dct(list(model.parameters())[0][0].view(_d,1)).squeeze().detach()
            
            
#             #### compute optimum learning rate:
# #             if _case == 1 and not optim_init:
# #                 x_tilde = dct(x).squeeze()
# #                 eta = 1/y.item()**2
# #                 opt = optim.SGD(model.parameters(), lr = eta)
# #                 optim_init = True
# #                 print("init")
# #             elif _case == 2 and not optim_init:
# #                 x_tilde = dct(x).squeeze()
# #                 eta = ((w_tilde_prev[0]-1)/(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + x_tilde[0]**2 * (w_tilde_prev[0]-1) )).item()
# #                 opt = optim.SGD(model.parameters(), lr = eta)
# #                 optim_init = True
# #                 print("init")
# #             elif _case == 3 and not optim_init:
                
#             if not optim_init:
#                 x_tilde = dct(x).squeeze()
#                 eta = ((w_tilde_prev[0]-1)/(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + x_tilde[0]**2 * (w_tilde_prev[0]-1) )).item()
#                 opt = optim.SGD(model.parameters(), lr = eta)
#                 optim_init = True
#                 print("init")
                

#             opt.zero_grad()

#             y_hat = model(x)
#             ipdb.set_trace()
# #             print(y_hat)

#             loss = ((1/2) * (y_hat.t() - y) ** 2).mean()
#             print("loss:", loss.item())
            
            
# #             print(w_tilde_prev - eta*(y_hat-y).item()*x_tilde)
#             loss_log[:,i] = loss.item()


#             loss.backward()
#             opt.step()

#             w[:,i] = list(model.parameters())[0][0].detach()
            
#             w_tilde[:,i] = dct(w[:,i].view(_d,1)).squeeze().detach()
# #             print(-eta*(x_tilde[0]*(x_tilde*w_tilde_prev)[1:].sum() + y.item()**2*(w_tilde_prev[0]-1)).item())
# #             ipdb.set_trace()
# #             print(-eta*(((x_tilde*w_tilde_prev).sum() - y.item())*x_tilde)[0].item())
#             print("gradient update at dim 0 : ", w_tilde[0,i] - w_tilde_prev[0])
#             print(w_tilde[0,i])
#             dw_tilde[0,:,i] = -(w_tilde[:,i] - w_tilde_prev).detach()
#             dw_tilde[1,:,i] = grad_estimation(w_tilde_prev, 
#                                               _lr, 
#                                               _case, 
#                                               _d, 
#                                               _mu, 
#                                               _std, 
#                                               _lambbda).squeeze().detach()

#         t.set_postfix(loss = loss.item())
#         t.update()
    
#     log_dict["w"] = w
#     log_dict["w_tilde"] = w_tilde
#     log_dict["dw_tilde"] = dw_tilde
#     log_dict["loss"] = loss_log

#     return log_dict

# def plot_first_dim_comparison_LR():
#     fig_test = plt.figure(figsize = [15,5])
#     gs = fig_test.add_gridspec(1,1)
#     p1 = fig_test.add_subplot(gs[0,0]).imshow(diff_map.cpu().detach().numpy(), cmap = 'Reds', aspect = 40.0, vmax = diff_map.max(), vmin = 0)
#     fig_test.colorbar(p1)
#     fig_test.add_subplot(gs[0,0]).set_ylabel("Frequency")
#     fig_test.add_subplot(gs[0,0]).set_xlabel("Training iteration")
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)