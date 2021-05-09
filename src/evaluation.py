import torch
import torch.nn as nn
from src.attacks import pgd_rand, pgd_rand_nn
from src.context import ctx_noparamgrad_and_eval
from src.utils_freq import rgb2gray, dct
import ipdb
from tqdm import trange

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    for x,y in loader:
        model.eval()
        if len(x.shape) ==2:
            x, y = x.t().to(device), y.t().to(device)
        else:
            x, y = x.to(device), y.to(device)

        with torch.no_grad():

            y_hat = model(x)
#             ipdb.set_trace()
            if len(loader.dataset.tensors[1].unique())==2:
                y = y.float().view(-1,1)
                loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                batch_correct = ((y_hat > 0) == (y==1)).sum().item()
            else:
                y = y.long()
                loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()
        
        total_correct += batch_correct
        total_loss += loss.item() * x.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def test_adv(loader, model, attack, param, device):
    total_loss, total_correct = 0.,0.
    total_delta_dct_2norm = 0.
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(model):
            delta = attack(**param).generate(model,X,y)

        delta_dct = dct(rgb2gray(delta), device)
        delta_dct_2norm = torch.norm(delta_dct, p = 2, dim = [1, 2])
        total_delta_dct_2norm += delta_dct_2norm.sum()

        with torch.no_grad():
            yp = model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
        
        total_correct += (yp.argmax(dim = 1) == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    delta_dct_2norm_mean = total_delta_dct_2norm / len(loader.dataset)
    return test_acc, test_loss, delta_dct_2norm_mean

def test_transfer_adv(loader, transferred_model, attacked_model, attack, param, device):
    total_loss, total_correct = 0.,0.
    for X,y in loader:
        transferred_model.eval()
        attacked_model.eval()
        X,y = X.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(transferred_model):
            delta = attack(**param).generate(transferred_model,X,y)
        with torch.no_grad():
            yp = attacked_model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
        
        total_correct += (yp.argmax(dim = 1) == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss




def test_pgd(loader, model, eps, device):
    total_correct = 0.
    
    param = {'ord': 2,
             'epsilon': eps,
             'alpha': 2.5*eps/100.,
             'num_iter': 100,
             'restarts': 1,
             'loss_fn': nn.CrossEntropyLoss()}
    
    if len(loader.dataset.tensors[1].unique())==2:
        param["loss_fn"] = torch.nn.BCEWithLogitsLoss()
        
    with trange(len(loader)) as t:
        for X,y in loader:
            model.eval()
            X,y = X.to(device), y.to(device)
            
            if len(loader.dataset.tensors[1].unique()) !=2:
                y = y.long()


            delta = pgd_rand_nn(**param).generate(model, X, y)
#             print(X.shape)
#             ipdb.set_trace()

            with torch.no_grad():
                y_hat = model(X+delta)
                if len(loader.dataset.tensors[1].unique())==2:
                    y = y.float().view(-1,1)
#                     loss = torch.nn.BCEWithLogitsLoss()(y_hat, y)
                    batch_correct = ((y_hat > 0) == (y==1)).sum().item()
                else:
                    y = y.long()
#                     loss = torch.nn.CrossEntropyLoss()(y_hat, y)
                    batch_correct = (y_hat.argmax(dim = 1) == y).sum().item()

            total_correct += batch_correct
#             total_loss += loss.item() * X.shape[0]
            
            t.set_postfix(acc = '{0:.2f}%'.format(batch_correct/X.shape[0]*100))
            t.update()
        
    test_acc = total_correct / len(loader.dataset) * 100
    return test_acc