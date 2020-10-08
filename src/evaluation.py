import torch
import torch.nn as nn
from src.attacks import pgd_rand
from src.context import ctx_noparamgrad_and_eval

def test_clean(loader, model, device):
    total_loss, total_correct = 0., 0.
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        with torch.no_grad():
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)
        
        total_correct += (yp.argmax(dim = 1) == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

def test_adv(loader, model, attack, param, device):
    total_loss, total_correct = 0.,0.
    for X,y in loader:
        model.eval()
        X,y = X.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(model):
            delta = attack(**param).generate(model,X,y)
        with torch.no_grad():
            yp = model(X+delta)
            loss = nn.CrossEntropyLoss()(yp,y)
        
        total_correct += (yp.argmax(dim = 1) == y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    test_acc = total_correct / len(loader.dataset) * 100
    test_loss = total_loss / len(loader.dataset)
    return test_acc, test_loss

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
