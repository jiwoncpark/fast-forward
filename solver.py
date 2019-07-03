import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from logger import Logger # for tensorboard logging

def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, n_epochs, device):
    print("Loading checkpoint at %s..." %checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.to(device)
    print("Starting with...")
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))
    return model, optimizer, epoch, loss

def fit_model(model, optimizer, lr_scheduler, n_epochs, train_loader, val_loader, 
    device, logging_interval, checkpointing_interval, X_val, Y_val, n_MC, run_id, checkpoint_path=None, verbose=True):
    n_train = train_loader.dataset.n_train
    n_val = val_loader.dataset.n_val

    if checkpoint_path is not None:
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, n_epochs, device)
        epoch += 1 # Advance one since last save
    else:
        epoch = 0
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logger = Logger('./logs')
    
    while epoch < n_epochs:
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        for X_, Y_ in train_loader:
            X_batch = Variable(torch.FloatTensor(X_)).to(device)
            Y_batch = Variable(torch.FloatTensor(Y_)).to(device)
            
            mean, logvar, regularization = model(X_batch)
            loss = nll_loss(Y_batch, mean, logvar) + regularization
            epoch_loss += loss.item()*X_batch.shape[0]/n_train
            
            optimizer.zero_grad()
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
        
        if (epoch+1)%(checkpointing_interval) == 0:    
            torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_cheduler': lr_scheduler.state_dict(),
            'loss': loss}, 'checkpoints/weights_%d_%d.pth' %(run_id, epoch+1))
            
        if (epoch+1)%(logging_interval) == 0:
            model.eval()
            with torch.no_grad():
                pppp, rmse = test(model, X_val, Y_val, n_MC, device)
                mean_norm = l2_norm(mean)
                logvar_norm = l2_norm(logvar)
                print('Epoch [{}/{}],\
                Loss: {:.4f}, PPPP: {:.2f}, RMSE: {:.4f}'.format(epoch+1, n_epochs, epoch_loss, pppp, rmse))
                # 1. Log scalar values (scalar summary)
                info = { 'loss': epoch_loss, 'PPPP': pppp, 'RMSE': rmse, 
                        'mean_norm': mean_norm, 'logvar_norm': logvar_norm }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch+1)

                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
            model.train()

        epoch += 1

    return model

def l2_norm(pred):
    norm_per_data = torch.norm(pred, dim=1) # shape [n_data,]
    return torch.mean(norm_per_data).item()

def nll_loss(true, mean, log_var):
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