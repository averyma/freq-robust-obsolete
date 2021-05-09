import torch
import torch.nn as nn
from src.attacks import pgd_rand, pgd_rand_nn
from src.context import ctx_noparamgrad_and_eval
from src.utils_freq import rgb2gray, dct, dct2, idct, idct2, batch_dct2, batch_idct2, getDCTmatrix
import ipdb
from tqdm import trange

def computeSensitivityMap(loader, model, eps, dim, numb_inputs, device):
    sens_map = torch.zeros(dim,dim,device=device)

    with trange(dim*dim) as t:
        for i in range(dim):
            for j in range(dim):
                sens_map[i,j] = test_freq_sensitivity(loader, model, eps, i,j, dim, numb_inputs, device)

                t.set_postfix(acc = '{0:.2f}%'.format(sens_map[i,j]))
                t.update()
    return sens_map

def avg_pgd_DCT(loader, model, eps, dim, device):
    dct_delta = torch.zeros(dim,dim, device=device)
    
    param = {'ord': 2,
             'epsilon': eps,
             'alpha': 2.5*eps/100.,
             'num_iter': 100,
             'restarts': 1,
             'loss_fn': nn.CrossEntropyLoss()}
    
    if len(loader.dataset.tensors[1].unique())==2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    dct_matrix = getDCTmatrix(dim)
    
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)
            
            if len(loader.dataset.tensors[1].unique())!=2:
                y = y.long()


            delta = pgd_rand_nn(**param).generate(model, X, y)
            dct_delta += batch_dct2(delta, dct_matrix).sum(dim=0)
            
            t.update()
        
    dct_delta = dct_delta / len(loader.dataset)
    return dct_delta

def single_image_freq_exam(loader, model, eps, device):
    
    
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        break
    
    
        
    param = {'ord': 2,
             'epsilon': eps,
             'alpha': 2.5*eps/100.,
             'num_iter': 100,
             'restarts': 1,
             'loss_fn': nn.CrossEntropyLoss()}
    
    if len(loader.dataset.tensors[1].unique())==2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    

    pgd = pgd_rand_nn(**param).generate(model, X, y)[0,:,:,:]
    X = X[0,:,:,:].view(1,1,X.shape[2],X.shape[3])
    y = y[0].view(1)
    
    dct_matrix = getDCTmatrix(X.shape[2])
    dct_pgd = batch_dct2(pgd, dct_matrix)
    dct_X = batch_dct2(X, dct_matrix)
    
    
    sens_map = torch.zeros(X.shape[2],X.shape[2], device = device)
    for i in range(X.shape[2]):
        for j in range(X.shape[3]):
            
            dct_delta = torch.zeros(1,1,X.shape[2],X.shape[2], device = device)
    
            dct_delta[0,0,i,j] = eps
            delta_pos = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
            dct_delta[0,0,i,j] = -eps
            delta_neg = idct2(dct_delta).view(1,1,X.shape[2],X.shape[2])
    
            model.eval()

            y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
            y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
                        
            if len(loader.dataset.tensors[1].unique())!=2:
                y = y.float().view(-1,1)
                pos_true = ((y_hat_pos > 0) == (y==1))
                neg_true = ((y_hat_neg > 0) == (y==1))
            else:
                y = y.long()
                pos_true = (y_hat_pos.argmax(dim = 1) == y)
                neg_true = (y_hat_neg.argmax(dim = 1) == y)
            
            sens_map[i,j] = pos_true*neg_true
    
    return X, dct_X, pgd, dct_pgd, sens_map

def avg_x_DCT(loader, dim, device):
    dct_matrix = getDCTmatrix(dim)
    dct_X = torch.zeros(dim,dim, device=device)
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        dct_X += batch_dct2(X, dct_matrix).sum(dim=0)
    
    dct_X = dct_X / len(loader.dataset)
    return dct_X.squeeze()


def test_freq_sensitivity(loader, model, eps, x, y, size, numb_inputs, device):
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
            
            y_hat_pos = model((X+delta_pos).clamp(min = 0, max = 1))
            y_hat_neg = model((X+delta_neg).clamp(min = 0, max = 1))
            
            if len(loader.dataset.tensors[1].unique())!=2:
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