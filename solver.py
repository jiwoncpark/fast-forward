import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from logger import Logger # for tensorboard logging

def load_checkpoint(model, optimizer, checkpoint_path):
    print("Loading checkpoint at %s..." %checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    return model, optimizer, epoch, loss

def fit_model(model, optimizer, n_epochs, train_loader, val_loader, n_val, device, logging_interval, X_val, Y_val, n_MC, run_id, checkpoint_path=None, verbose=True):
    if checkpoint_path is not None:
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        epoch = 0

    logger = Logger('./logs')
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    while epoch < n_epochs:
        for X_, Y_ in train_loader:
            X_batch = Variable(torch.FloatTensor(X_)).to(device)
            Y_batch = Variable(torch.FloatTensor(Y_)).to(device)
            
            mean, logvar, regularization = model(X_batch)
            loss = heteroscedastic_loss(Y_batch, mean, logvar) + regularization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1)%(logging_interval) == 0:
            
            with torch.no_grad():
                pppp, rmse = test(model, X_val, Y_val, n_MC, device)
                mean_norm = l2_norm(mean)
                logvar_norm = l2_norm(logvar)
                print('Epoch [{}/{}],\
                Loss: {:.4f}, PPPP: {:.2f}, RMSE: {:.2f}'.format(epoch+1, n_epochs, loss.item(), pppp, rmse))
                
                # 1. Log scalar values (scalar summary)
                info = { 'loss': loss.item(), 'PPPP': pppp, 'RMSE': rmse, 
                        'mean_norm': mean_norm, 'logvar_norm': logvar_norm }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        epoch += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, 'checkpoints/weights_%d.pth' %run_id)

    return model

def l2_norm(pred):
    norm_per_data = torch.norm(pred, dim=1) # shape [n_data,]
    return torch.mean(norm_per_data).item()

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
                 ~= log 1/n_MC sum p(y|x, w_k) with w_k sim q(w)
                  = LogSumExp log p(y|x, w_k) - log n_MC
    :Y_true: a 2D array of size N x Y_dim

    Note
    ----
    Does not use torch
    """
    model.eval()
    n_val, Y_dim = Y_val.shape
    MC_samples = [model(Variable(torch.FloatTensor(X_val)).to(device)) for _ in range(n_MC)] # shape [K, N, 2D]
    means = torch.stack([tup[0] for tup in MC_samples]).view(n_MC, n_val, Y_dim).cpu().data.numpy()
    logvar = torch.stack([tup[1] for tup in MC_samples]).view(n_MC, n_val, Y_dim).cpu().data.numpy()

    # per point predictive probability
    test_ll = -0.5*np.exp(-logvar)*(means - Y_val.squeeze())**2.0 - 0.5*logvar - 0.5*np.log(2.0*np.pi) # shape [K, N, D]
    test_ll = np.sum(np.sum(test_ll, axis=-1), axis=-1) # shape [K,]
    test_ll = logsumexp(test_ll) - np.log(n_MC)
    pppp = test_ll/n_val # FIXME: not sure why we don't do - np.log(n_val) instead 

    # root mean-squared error
    rmse = np.mean( (np.mean(means, axis=0) - Y_val.squeeze())**2.0 )
    return pppp, rmse

if __name__ == '__main__':
    pass