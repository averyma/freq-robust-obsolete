import torch
import torch.nn as nn
from src.attacks import pgd_rand, pgd_rand_nn, fgsm_nn
from src.context import ctx_noparamgrad_and_eval
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, batch_idct2, getDCTmatrix
import ipdb
from tqdm import trange

def computeSensitivityMap(loader, model, eps, dim, numb_inputs, clip, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity(loader, model, eps, i,j, dim, numb_inputs, clip, device)

                t.set_postfix(acc = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def avg_attack_DCT(loader, model, eps, dim, attack, device):
    dct_delta = torch.zeros(dim,dim, device=device)
    
    if attack == "pgd":
        param = {'ord': 2,
                 'epsilon': eps,
                 'alpha': 2.5*eps/100.,
                 'num_iter': 100,
                 'restarts': 1,
                 'loss_fn': nn.CrossEntropyLoss()}
    elif attack == "fgsm":
        param = {'ord': 2,
             'epsilon': eps,
             'loss_fn': nn.CrossEntropyLoss()}
        
    
    if len(loader.dataset.classes) == 2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    dct_matrix = getDCTmatrix(dim)
    
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)
            
            if len(loader.dataset.classes) != 2:
                y = y.long()

            if attack == "pgd":
                delta = pgd_rand_nn(**param).generate(model, X, y)
            elif attack == "fgsm":
                delta = fgsm_nn(**param).generate(model, X, y)
                
            dct_delta += batch_dct2(delta, dct_matrix).sum(dim=0)
            
            t.update()
        
    dct_delta = dct_delta / len(loader.dataset)
    return dct_delta

def single_image_freq_exam(loader, model, eps, attack, clip, device):
    
    
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        break
    
    
    
    if attack == "pgd":
        param = {'ord': 2,
                 'epsilon': eps,
                 'alpha': 2.5*eps/100.,
                 'num_iter': 100,
                 'restarts': 1,
                 'loss_fn': nn.CrossEntropyLoss()}
    elif attack == "fgsm":
        param = {'ord': 2,
                 'epsilon': eps,
                 'loss_fn': nn.CrossEntropyLoss()}
    
    if len(loader.dataset.classes) == 2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    
    if attack == "pgd":
        attack_delta = pgd_rand_nn(**param).generate(model, X, y)[0,:,:,:]
    elif attack == "fgsm":
        attack_delta = fgsm_nn(**param).generate(model, X, y)[0,:,:,:]
                
    X = X[0,:,:,:].view(1,1,X.shape[2],X.shape[3])
    y = y[0].view(1)
    
    dct_matrix = getDCTmatrix(X.shape[2])
    dct_attack_delta = batch_dct2(attack_delta, dct_matrix)
    dct_X = batch_dct2(X, dct_matrix)
    
    
    sens_map = torch.zeros(X.shape[2],X.shape[2], device = device)
    for i in range(X.shape[2]):
        for j in range(X.shape[3]):
            
            dct_delta = torch.zeros(1,1,X.shape[2],X.shape[2], device = device)
    
            dct_delta[0,0,i,j] = eps
            delta_pos = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
            dct_delta[0,0,i,j] = -eps
            delta_neg = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
#             ipdb.set_trace()
            model.eval()
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos))
                y_hat_neg = model((X+delta_neg))
                        
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                pos_true = ((y_hat_pos > 0) == (y==1))
                neg_true = ((y_hat_neg > 0) == (y==1))
            else:
                y = y.long()
                pos_true = (y_hat_pos.argmax(dim = 1) == y)
                neg_true = (y_hat_neg.argmax(dim = 1) == y)
#             ipdb.set_trace()
            sens_map[i,j] = pos_true*neg_true
    
    eps_map = torch.zeros(X.shape[2],X.shape[2], device = device)
    with trange(X.shape[2]**2) as t:
        for i in range(X.shape[2]):
            for j in range(X.shape[3]):
                k = 0
                neg_true = 1
                pos_true = 1
                while neg_true ==1 and pos_true==1: #both correct
                    k += 1
                    dct_delta = torch.zeros(1,1,X.shape[2],X.shape[2], device = device)

                    dct_delta[0,0,i,j] = k
                    delta_pos = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
                    dct_delta[0,0,i,j] = -k
                    delta_neg = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])

    #                 ipdb.set_trace()
                    model.eval()
                    if clip:
                        y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                        y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
                    else:
                        y_hat_pos = model(X+delta_pos)
                        y_hat_neg = model(X+delta_neg)

                    if len(loader.dataset.classes) == 2:
                        y = y.float().view(-1,1)
                        pos_true = ((y_hat_pos > 0) == (y==1)).sum().item()
                        neg_true = ((y_hat_neg > 0) == (y==1)).sum().item()
                    else:
                        y = y.long()
                        pos_true = (y_hat_pos.argmax(dim = 1) == y).sum().item()
                        neg_true = (y_hat_neg.argmax(dim = 1) == y).sum().item()
                        
                    t.set_postfix(i = '{0:.2f}'.format(i),
                                  j = '{0:.2f}'.format(j),
                                  k = '{0:.2f}'.format(k))
                
                    
                    if k ==100:
                        break
                t.update()


                eps_map[i,j] = k
    
    return X, dct_X, attack_delta, dct_attack_delta, sens_map, eps_map

def avg_x_DCT(loader, dim, device):
    dct_matrix = getDCTmatrix(dim)
    dct_X = torch.zeros(dim,dim, device=device)
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        dct_X += batch_dct2(X, dct_matrix).sum(dim=0)
    
    dct_X = dct_X / len(loader.dataset)
    return dct_X.squeeze()


def test_freq_sensitivity(loader, model, eps, x, y, size, numb_inputs, clip, device):
    total_correct = 0.

    dct_delta = torch.zeros(1,1,size,size, device = device)
    
    dct_delta[0,0,x,y] = eps
    delta_pos = idct2(dct_delta).view(1,1,size,size)
    dct_delta[0,0,x,y] = -eps
    delta_neg = idct2(dct_delta).view(1,1,size,size)
    
    model.eval()
    total_tested_input = 0 
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            
            if clip:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            else:
                y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
                y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            
            if len(loader.dataset.classes) == 2:
                y = y.float().view(-1,1)
                batch_correct_pos = ((y_hat_pos > 0) == (y==1)).sum().item()
                batch_correct_neg = ((y_hat_neg > 0) == (y==1)).sum().item()
            else:
                y = y.long()
                batch_correct_pos = (y_hat_pos.argmax(dim = 1) == y).sum().item()
                batch_correct_neg = (y_hat_neg.argmax(dim = 1) == y).sum().item()
                
        if batch_correct_pos > batch_correct_neg:
            batch_correct = batch_correct_neg
        else:
            batch_correct = batch_correct_pos
        
        total_correct += batch_correct
        
        total_tested_input += X.shape[0]
        if total_tested_input>numb_inputs:
            break
        
    test_acc = total_correct / total_tested_input * 100
    return test_acc