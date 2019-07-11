import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from logger import Logger # for tensorboard logging
import plotting_utils as plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import uncertainty_utils as uncertain
from collections import OrderedDict
import io

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

def fit_model(model, optimizer, lr_scheduler, train_loader, val_loader, 
    device, args, data_meta, X_val, Y_val, checkpoint_path=None):
    n_train = len(data_meta['train_indices'])
    n_val = len(data_meta['val_indices'])
    val_sampled_good = [5182, 5208,  166, 6136, 3789, 1092, 6300, 3729, 6145,  258, 4318,
                   3006, 3917, 3206,  557, 2977, 4458, 6104, 2923, 3300, 3674,  734,
                   2997, 4865, 3988, 2008, 2031, 4745, 1259, 2730,  689, 2277, 4363,
                   2904, 3881, 2629, 4995, 5171, 4533, 5032, 4682, 2632, 2004, 4116,
                   6425, 6420, 4946, 5316, 5343, 2037, 1721,  616, 5492, 3975, 6188,
                   4107, 4416, 6157, 6700, 5909, 4529, 6511, 2582, 2823, 6229, 3629,
                   1722, 2627,  309, 3595, 2235, 5919, 1305, 3839, 6212, 2446, 4328,
                   3930, 4469,  456, 1377,  970, 5702, 4866, 4678, 3438, 5707, 1415,
                   3237, 3738, 5358, 5600, 1821, 3452, 6207, 5619,  378, 5929, 5928,
                   3647,  405, 2581, 2777, 3714, 6650,  403, 3573, 4110, 2386, 2196,
                   5579, 5698, 4896, 5373, 6006, 3520, 6560, 1900, 3797, 4709, 2041,
                   5416, 3733, 5741, 1957, 6355, 2973, 2070, 4918, 1947, 1242,  736,
                   5783, 4433, 5295,  949, 1258, 4196, 4445, 3687,  223, 3916, 2811,
                   3689, 6513, 3791, 5197, 5297, 5901, 4642, 5984, 2510, 5948,  695,
                     89, 6694, 2588, 3784, 6443,  404, 3437, 1027, 3243, 5103, 4150,
                   1373, 6618,  626, 3800, 1904, 3459,  794, 1634,  612, 5408, 6211,
                   1261, 3987, 2222, 5757, 1911, 2875, 2667, 5283, 3644, 5061, 4942,
                   6574, 6600, 3519, 6611, 2796, 6717, 1427,  509,  926, 1475, 2612,
                   5540, 3333]
    val_sampled_bad = [6518, 1300, 1309, 2134, 4271,  328, 4949,  989,  114, 4614, 3999,
                   4123, 5534, 3487,  290, 5782, 5260, 3012, 4186,  148, 2036, 2035,
                   4643, 1272, 2463, 5684, 1485, 2607, 1571, 6580, 5154,  228,  136,
                   3544, 5791, 1783, 6159, 6007, 6235,  744, 6566, 1813,  937, 5415,
                    624, 2506, 4460, 5383, 1187,  663]
    val_sampled = val_sampled_good + val_sampled_bad
    X_val_sampled = X_val[val_sampled, :] # shape [n_subsampled, X_dim]
    Y_val_sampled = Y_val[val_sampled, :] # shape [n_subsampled, Y_dim]


    if checkpoint_path is not None:
        model, optimizer, epoch, loss = load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path, args['n_epochs'], device)
        epoch += 1 # Advance one since last save
    else:
        epoch = 0
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
        
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logger = Logger('./logs')
    
    while epoch < args['n_epochs']:
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        for X_, Y_ in train_loader:
            X_batch = Variable(torch.FloatTensor(X_)).to(device)
            Y_batch = Variable(torch.FloatTensor(Y_)).to(device)
            
            mean, logvar, mean_classifier, logvar_classifier, regularization = model(X_batch)
            # regression loss
            loss = nll_loss_regress(Y_batch[:, 1:], mean, logvar) + regularization
            # classification loss
            loss += nll_loss_classify(Y_batch[:, 0].view([-1, 1]), mean_classifier, logvar_classifier)
            
            epoch_loss += loss.item()*X_batch.shape[0]/n_train
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1)%(args['checkpointing_interval']) == 0:    
            torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': loss}, 'checkpoints/weights_%d_%d.pth' %(args['run_id'], epoch+1))
            
        if (epoch+1)%(args['logging_interval']) == 0:
            model.eval()
            with torch.no_grad():
                means, logvars, means_class, logvars_class = mc_sample(model, X_val, Y_val, args['n_MC'], device)
                pppp, rmse, mean_norm, logvar_norm = get_scalar_metrics(means, logvars, Y_val[:, 1:], args['n_MC'])
                
                print('Epoch [{}/{}],\
                Loss: {:.4f}, PPPP: {:.2f}, RMSE: {:.4f}'.format(epoch+1, args['n_epochs'], epoch_loss, pppp, rmse))
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

                # 3. Log training images (image summary)
                mu = np.mean(means, axis=0)
                al_sig2 = uncertain.get_aleatoric_sigma2(logvars)
                ep_sig2 = uncertain.get_epistemic_sigma2(means)
                
                mu_class = np.mean(1.0 / (1.0 + np.exp(-means_class)), axis=0)
                al_sig2_class = uncertain.get_aleatoric_sigma2(logvars_class)
                ep_sig2_class = uncertain.get_epistemic_sigma2(means_class)
                # Convert to natural units
                X_to_plot, Y_to_plot, em_to_plot = plotting.get_natural_units(X_val_sampled, Y_val_sampled,
                    mu[val_sampled, :], al_sig2[val_sampled, :],
                    ep_sig2[val_sampled, :], mu_class[val_sampled, :],
                    al_sig2_class[val_sampled, :], ep_sig2_class[val_sampled, :],
                    data_meta)
                X_full, Y_full, em_full = plotting.get_natural_units(X_val, Y_val,
                    mu, al_sig2, ep_sig2, mu_class, al_sig2_class, ep_sig2_class, data_meta)

                # Get mapping plots
                psFlux_mag = get_magnitude_plot(epoch+1, X_to_plot.loc[:200, :], Y_to_plot.loc[:200, :], em_to_plot.loc[:200, :], 'psFlux_%s', data_meta)
                cModelFlux_mag = get_magnitude_plot(epoch+1, X_to_plot.loc[:200, :], Y_to_plot.loc[:200, :], em_to_plot.loc[:200, :], 'cModelFlux_%s', data_meta)
                psFlux = get_flux_plot(epoch+1, X_to_plot, Y_to_plot, em_to_plot, 'psFlux_%s', data_meta)
                cModelFlux = get_flux_plot(epoch+1, X_to_plot, Y_to_plot, em_to_plot, 'cModelFlux_%s', data_meta)
                moments = get_moment_plot(epoch+1, X_to_plot, Y_to_plot, em_to_plot)
                conf_mat = get_star_metrics(epoch+1, X_full, Y_full, em_full)

                info = {
                'psFlux_mapping (mag)': psFlux_mag,
                'cModelFlux_mapping (mag)': cModelFlux_mag,
                'psFlux_mapping (Jy)': psFlux,
                'cModelFlux_mapping (Jy)': cModelFlux,
                'moments': moments,
                'star classification': conf_mat}                 

                for tag, images in info.items():
                    logger.image_summary(tag, images, epoch+1)

            model.train()

        epoch += 1
        lr_scheduler.step()

    return model

def get_star_metrics(epoch, X, Y, emulated):
    my_dpi = 72.0
    fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
    canvas = plotting.plot_confusion_matrix(fig, X, Y, emulated)
    width, height = fig.get_size_inches() * fig.get_dpi()
    conf_mat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
    return conf_mat

def get_moment_plot(epoch, X, Y, emulated):
    my_dpi = 72.0
    per_filter = []
    for moment_type in ['Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF', 'ra_offset', 'dec_offset']:
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_moment(fig, X, Y, emulated, moment_type)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def get_flux_plot(epoch, X, Y, emulated, flux_formatting, data_meta):
    my_dpi = 72.0
    per_filter = []
    for bp in 'ugrizy':
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_flux(fig, X, Y, emulated, flux_formatting, bp)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def get_magnitude_plot(epoch, X, Y, emulated, flux_formatting, data_meta):
    my_dpi = 72.0
    per_filter = []
    for bp in 'ugrizy':
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_magnitude(fig, X, Y, emulated, flux_formatting, bp)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def l2_norm(pred):
    norm_per_data = np.linalg.norm(pred, axis=2) # shape [n_MC, n_data]
    return np.mean(norm_per_data)

def nll_loss_regress(true, mean, log_var):
    precision = torch.exp(-log_var)
    return torch.mean(torch.sum(precision * (true - mean)**2.0 + log_var, dim=1), dim=0)

def nll_loss_classify(true, mean, log_var):
    #precision = torch.exp(-log_var)
    #return torch.mean(torch.sum(precision * (true - mean)**2.0 + log_var, dim=1), dim=0)
    loss = nn.BCEWithLogitsLoss()
    return loss(mean, true)

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def mc_sample(model, X_val, Y_val, n_MC, device):
    n_val, Y_dim = Y_val.shape
    MC_samples = [model(Variable(torch.FloatTensor(X_val)).to(device)) for _ in range(n_MC)] # shape [K, N, 2D]
    means = torch.stack([tup[0] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy()
    logvars = torch.stack([tup[1] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy()
    means_class = torch.stack([tup[2] for tup in MC_samples]).view(n_MC, n_val, 1).cpu().data.numpy()
    logvars_class = torch.stack([tup[3] for tup in MC_samples]).view(n_MC, n_val, 1).cpu().data.numpy()
    return means, logvars, means_class, logvars_class

def get_scalar_metrics(means, logvar, Y_val, n_MC):
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
    # per point predictive probability
    n_val, Y_dim = Y_val.shape
    test_ll = -0.5*np.exp(-logvar)*(means - Y_val.squeeze())**2.0 - 0.5*logvar - 0.5*np.log(2.0*np.pi) # shape [K, N, D]
    test_ll = np.sum(np.sum(test_ll, axis=-1), axis=-1) # shape [K,]
    test_ll = logsumexp(test_ll) - np.log(n_MC)
    pppp = test_ll/n_val # FIXME: not sure why we don't do - np.log(n_val) instead 

    # root mean-squared error
    rmse = np.mean( (np.mean(means, axis=0) - Y_val.squeeze())**2.0 )
    return pppp, rmse, l2_norm(means), l2_norm(logvar)

if __name__ == '__main__':
    pass