import torch
import torch.nn as nn

from tqdm import trange
import numpy as np

from src.attacks import pgd_rand
from src.context import ctx_noparamgrad_and_eval
from src.utils_general import ep2itr

def data_init(init, X, y, model):
    if init == "rand":
        delta = torch.empty_like(X.detach(), requires_grad=False).uniform_(-8./255.,8./255.)
        delta.data = (X.detach() + delta.detach()).clamp(min = 0, max = 1.0) - X.detach()
    elif init == "fgsm":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 2./255.}
            delta = fgsm(**param).generate(model,X,y)
    elif init == "pgd1":
        with ctx_noparamgrad_and_eval(model):
            param = {"ord":np.inf, "epsilon": 8./255., "alpha":2./255., "num_iter": 1, "restart": 1}
            delta = pgd_rand(**param).generate(model,X,y)
    elif init == "none":
        delta = torch.zeros_like(X.detach(), requires_grad=False)

    return delta

def train_standard(logger, epoch, loader, model, opt, device):
    total_loss, total_correct = 0., 0.
    curr_itr = ep2itr(epoch, loader)
    with trange(len(loader)) as t:
        for X, y in loader:
            model.train()
            X, y = X.to(device), y.to(device)

            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_correct = (yp.argmax(dim=1) == y).sum().item()
            total_correct += batch_correct

            batch_acc = batch_correct / X.shape[0]
            total_loss += loss.item() * X.shape[0]

            t.set_postfix(loss=loss.item(),
                          acc='{0:.2f}%'.format(batch_acc*100))
            t.update()
            curr_itr += 1
            logger.add_scalar("train/acc_itr", batch_acc, curr_itr)
            logger.add_scalar("train/loss_itr", loss, curr_itr)

    acc = total_correct / len(loader.dataset) * 100
    total_loss = total_loss / len(loader.dataset)

    return acc, total_loss

def train_adv(logger, epoch, loader, pgd_steps, model, opt, device):
    total_loss_adv = 0.
    total_correct_adv = 0.

    attack = pgd_rand
    curr_itr = ep2itr(epoch, loader)
    param = {'ord': np.inf,
             'epsilon': 8./255.,
             'alpha': 2./255.,
             'num_iter': pgd_steps,
             'restarts': 1}

    with trange(len(loader)) as t:
        for X,y in loader:
            model.train()
            X,y = X.to(device), y.to(device)

            with ctx_noparamgrad_and_eval(model):
                delta = attack(**param).generate(model, X, y)
            
            yp_adv = model(X+delta)
            loss_adv = nn.CrossEntropyLoss()(yp_adv, y)
                
            opt.zero_grad()
            loss_adv.backward()
            opt.step()
    
            batch_correct_adv = (yp_adv.argmax(dim = 1) == y).sum().item()
            total_correct_adv += batch_correct_adv

            batch_acc_adv = batch_correct_adv / X.shape[0]
            total_loss_adv += loss_adv.item() * X.shape[0]

            t.set_postfix(loss_adv = loss_adv.item(),
                          acc_adv = '{0:.2f}%'.format(batch_acc_adv*100))
            t.update()
            curr_itr += 1
            logger.add_scalar("train/acc_itr", batch_acc_adv, curr_itr)
            logger.add_scalar("train/loss_itr", loss_adv, curr_itr)

    acc_adv = total_correct_adv / len(loader.dataset) * 100
    total_loss_adv = total_loss_adv / len(loader.dataset)

    return acc_adv, total_loss_adv
