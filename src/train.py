import torch
import torch.nn as nn

from tqdm import trange
import numpy as np

from src.attacks import pgd_rand
from src.context import ctx_noparamgrad_and_eval
from src.utils_general import ep2itr
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct, batch_dct2, getDCTmatrix
import ipdb

from collections import defaultdict

# def data_init(init, X, y, model):
#     if init == "rand":
#         delta = torch.empty_like(X.detach(), requires_grad=False).uniform_(-8./255.,8./255.)
#         delta.data = (X.detach() + delta.detach()).clamp(min = 0, max = 1.0) - X.detach()
#     elif init == "fgsm":
#         with ctx_noparamgrad_and_eval(model):
#             param = {"ord":np.inf, "epsilon": 2./255.}
#             delta = fgsm(**param).generate(model,X,y)
#     elif init == "pgd1":
#         with ctx_noparamgrad_and_eval(model):
#             param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 1, "restart": 1}
#             delta = pgd_rand(**param).generate(model,X,y)
#     elif init == "none":
#         delta = torch.zeros_like(X.detach(), requires_grad=False)

#     return delta

# def train_standard(logger, epoch, loader, model, opt, device):
    # total_loss, total_correct = 0., 0.
    # curr_itr = ep2itr(epoch, loader)
    # with trange(len(loader)) as t:
        # for X, y in loader:
            # model.train()
            # X, y = X.to(device), y.to(device)

            # yp = model(X)
            # loss = nn.CrossEntropyLoss()(yp, y)

            # opt.zero_grad()
            # loss.backward()
            # opt.step()

            # batch_correct = (yp.argmax(dim=1) == y).sum().item()
            # total_correct += batch_correct

            # batch_acc = batch_correct / X.shape[0]
            # total_loss += loss.item() * X.shape[0]

            # t.set_postfix(loss=loss.item(),
                          # acc='{0:.2f}%'.format(batch_acc*100))
            # t.update()
            # curr_itr += 1
            # logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
            # logger.add_scalar("train/loss_itr", loss, curr_itr)

    # acc = total_correct / len(loader.dataset) * 100
    # total_loss = total_loss / len(loader.dataset)

    # return acc, total_loss

# def train_standard(logger, epoch, loader, model, opt, device):
#     total_loss_adv = 0.
#     total_correct_adv = 0.

#     attack = pgd_rand
#     curr_itr = ep2itr(epoch, loader)
#     param = {'ord': np.inf,
#              'epsilon': 8./255.,
#              'alpha': 2./255.,
#              'num_iter': 10,
#              'restarts': 1}

#     with trange(len(loader)) as t:
#         for X,y in loader:
#             model.train()
#             X,y = X.to(device), y.to(device)

#             with ctx_noparamgrad_and_eval(model):
#                 delta = attack(**param).generate(model, X, y)
        
#             delta_dct = dct(rgb2gray(delta), device)
#             delta_dct_2norm = torch.norm(delta_dct, p = 2, dim = [1, 2]).mean()

#             yp_adv = model(X)
#             loss_adv = nn.CrossEntropyLoss()(yp_adv, y)
                
#             opt.zero_grad()
#             loss_adv.backward()
#             opt.step()
    
#             batch_correct_adv = (yp_adv.argmax(dim = 1) == y).sum().item()
#             total_correct_adv += batch_correct_adv

#             batch_acc_adv = batch_correct_adv / X.shape[0]
#             total_loss_adv += loss_adv.item() * X.shape[0]

#             t.set_postfix(loss_adv = loss_adv.item(),
#                           acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
#             t.update()
#             curr_itr += 1
#             logger.add_scalar("train/acc_itr", batch_acc_adv, curr_itr)
#             logger.add_scalar("train/loss_itr", loss_adv, curr_itr)
#             logger.add_scalar("train/dct_2norm", delta_dct_2norm, curr_itr)

#     acc_adv = total_correct_adv / len(loader.dataset) * 100
#     total_loss_adv = total_loss_adv / len(loader.dataset)

#     return acc_adv, total_loss_adv

# def train_adv(logger, epoch, loader, pgd_steps, model, opt, device):
#     total_loss_adv = 0.
#     total_correct_adv = 0.

#     attack = pgd_rand
#     curr_itr = ep2itr(epoch, loader)
#     param = {'ord': np.inf,
#              'epsilon': 8./255.,
#              'alpha': 2./255.,
#              'num_iter': pgd_steps,
#              'restarts': 1}

#     with trange(len(loader)) as t:
#         for X,y in loader:
#             model.train()
#             X,y = X.to(device), y.to(device)

#             with ctx_noparamgrad_and_eval(model):
#                 delta = attack(**param).generate(model, X, y)
        
#             delta_dct = dct(rgb2gray(delta), device)
#             delta_dct_2norm = torch.norm(delta_dct, p = 2, dim = [1, 2]).mean()

#             yp_adv = model(X+delta)
#             loss_adv = nn.CrossEntropyLoss()(yp_adv, y)
                
#             opt.zero_grad()
#             loss_adv.backward()
#             opt.step()
    
#             batch_correct_adv = (yp_adv.argmax(dim = 1) == y).sum().item()
#             total_correct_adv += batch_correct_adv

#             batch_acc_adv = batch_correct_adv / X.shape[0]
#             total_loss_adv += loss_adv.item() * X.shape[0]

#             t.set_postfix(loss_adv = loss_adv.item(),
#                           acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
#             t.update()
#             curr_itr += 1
#             logger.add_scalar("train/acc_itr", batch_acc_adv, curr_itr)
#             logger.add_scalar("train/loss_itr", loss_adv, curr_itr)
#             logger.add_scalar("train/dct_2norm", delta_dct_2norm, curr_itr)

#     acc_adv = total_correct_adv / len(loader.dataset) * 100
#     total_loss_adv = total_loss_adv / len(loader.dataset)

#     return acc_adv, total_loss_adv


def train_NN_synthetic(loader, args, model, opt, log_theta_tilde, device):
    # this is for training case 1,2,3,4,5,6
    
    _case = args["case"]
    _iteration = args["itr"]
    _input_d = args["input_d"]
    _output_d = args["output_d"]
    _hidden_d = args["hidden_d"]
    
    log_dict = defaultdict(lambda: list())
    
    if log_theta_tilde:
        numb_theta_logged = 5
        theta_tilde_logger = torch.zeros(numb_theta_logged, _input_d, _iteration, device = device)
        dct_matrix = getDCTmatrix(_input_d).to(device)
        random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                                
                if log_theta_tilde:
                    theta = model.state_dict()["linear1.weight"].detach()[random_theta_m_index,:]
                    theta_tilde_logger[:,:,i] =  batch_dct(theta, dct_matrix)
                    
                x, y = x.t().to(device), y.t().to(device)
                x_len = x.shape[1]

                opt.zero_grad()
                
                y_hat = model(x)

                if _output_d == 1:
                    y = y.float().view(-1,1)
                    loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                else:
                    y = y.long()
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                acc_logger[:,i] = batch_acc

                loss.backward()
                opt.step()

                i += 1
                
                t.set_postfix(loss = loss.item(),
                              acc = '{0:.2f}%'.format(batch_acc))
                t.update()
                if i == _iteration:
                    break
    
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    if log_theta_tilde:
        log_dict["theta_tilde"] = theta_tilde_logger
    
    return log_dict

def train_NN_real(loader, args, model, opt, log_theta_tilde, device):
    
    _case = args["case"]
    _iteration = args["itr"]
    _output_d = args["output_d"]
    _input_d = args["input_d"]
    _hidden_d = args["hidden_d"]
    _eps = args["eps"]
    
#     param = {'ord': 2,
#              'epsilon': _eps,
#              'alpha': 2.5*_eps/100.,
#              'num_iter': 50,
#              'restarts': 1,
#              'loss_fn': nn.CrossEntropyLoss()}
#     if _case in [4,12]:
#         param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
    
    log_dict = defaultdict(lambda: list())
    
    if log_theta_tilde:
        numb_theta_logged = 5
        theta_tilde_logger = torch.zeros(numb_theta_logged, _input_d, _input_d, _iteration)
        dct_matrix = getDCTmatrix(_input_d).to(device)
        random_theta_m_index = torch.randint(low=0, high = _hidden_d, size = (numb_theta_logged,))
    
    loss_logger = torch.zeros(1, _iteration, device = device)
    acc_logger = torch.zeros(1, _iteration, device = device)
    
    i = 0
    with trange(_iteration) as t:
        while(i < _iteration):
            for x, y in loader:
                
                if log_theta_tilde:
#                     ipdb.set_trace()
                    theta = model.state_dict()["conv1.weight"].detach()[random_theta_m_index,:,:,:]
                    theta_tilde_logger[:,:,:,i] =  batch_dct2(theta, dct_matrix)

                x, y = x.to(device), y.to(device)
                x_len = x.shape[0]
                
                
                opt.zero_grad()
                
#                 if adv: 
#                     delta = pgd_rand_nn(**param).generate(model, x, y)
#                     y_hat = model(x+delta)
#                 else:
                y_hat = model(x)

                if _output_d == 1:
                    y = y.float().view(-1,1)
                    loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                    
                else:
                    y = y.long()
                    loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

                loss_logger[:,i] = loss.item()
                batch_acc = batch_correct / x_len * 100
                acc_logger[:,i] = batch_acc

                loss.backward()
                opt.step()

                
                i += 1
                
                t.set_postfix(loss = loss.item(),
                              acc = '{0:.2f}%'.format(batch_acc))
                t.update()
                if i == _iteration:
                    break
    
    log_dict["loss"] = loss_logger
    log_dict["acc"] = acc_logger
    if log_theta_tilde:
        log_dict["theta_tilde"] = theta_tilde_logger
    
    return log_dict