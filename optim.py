import os
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

def checkpoint_val(model, val_loader, n_val, device):
    val_loss = 0
    for X_val_, Y_val_ in val_loader:
        X_val_batch = Variable(torch.FloatTensor(X_val_)).to(device)
        Y_val_batch = Variable(torch.FloatTensor(Y_val_)).to(device)
        
        mean, log_var, regularization = model(X_val_batch)
        val_batch_size = Y_val_.shape[0]
        batch_loss = heteroscedastic_loss(Y_val_batch, mean, log_var) + regularization
        val_loss += val_batch_size*batch_loss
    val_loss /= n_val
    return val_loss

def fit_model(model, n_epochs, train_loader, val_loader, n_val, device, logging_interval, X_val, Y_val, n_MC, run_id, verbose=True):
    model = model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=5e-4)
    
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    pppp = [] # per point predictive probability
    rmse = []
    
    for i in range(n_epochs):
        for X_, Y_ in train_loader:
            X_batch = Variable(torch.FloatTensor(X_)).to(device)
            Y_batch = Variable(torch.FloatTensor(Y_)).to(device)
            
            mean, log_var, regularization = model(X_batch)
            loss = heteroscedastic_loss(Y_batch, mean, log_var) + regularization
            
            optimizer.zero_grad()
            loss.backward()
            
            #clip_grad_norm_(model.parameters(), 0.02)
            #for p in model.parameters():
            #    p.data.add_(-1e-4, p.grad.data)
            
            optimizer.step()
        
        if (i+1)%(logging_interval) == 0:
            with torch.no_grad():
                p, r = test(model, X_val, Y_val, n_MC, device)
                pppp.append(p)
                rmse.append(r)
            print("Epoch %d done" %(i+1))
            
    torch.save(model.state_dict(), 'checkpoint/weights_%d.pth' %run_id)

    return model, pppp, rmse

def heteroscedastic_loss(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean)**2.0 + log_var, dim=1), dim=0)

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def test(model, X_val, Y_val, n_MC, device):
    """
    Estimate predictive log likelihood:
    log p(y|x, D) = log int p(y|x, w) p(w|D) dw
                 ~= log int p(y|x, w) q(w) dw
                 ~= log 1/K sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log K
    :Y_true: a 2D array of size N x dim
    :MC_samples: a 3D array of size samples K x N x 2*D
    """
    model.eval()
    n_val, Y_dim = Y_val.shape
    MC_samples = [model(Variable(torch.FloatTensor(X_val)).to(device)) for _ in range(n_MC)]
    means = torch.stack([tup[0] for tup in MC_samples]).view(n_MC, n_val, Y_dim).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(n_MC, n_val, Y_dim).cpu().data.numpy()
    
    test_ll = -0.5*np.exp(-logvar)*(means - Y_val.squeeze())**2.0 - 0.5*logvar - 0.5*np.log(2.0*np.pi) #Y_true[None] # shape [K, N, D]
    test_ll = np.sum(np.sum(test_ll, -1), -1) # shape [K,]
    test_ll = logsumexp(test_ll) - np.log(n_MC)
    pppp = test_ll/n_val  # per point predictive probability
    rmse = np.mean((np.mean(means, 0) - Y_val.squeeze())**2.)**0.5
    return pppp, rmse