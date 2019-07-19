import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
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
            
            mean, logvar, F, mean2, logvar2, F2, alpha, mean_classifier, logvar_classifier, regularization = model(X_batch)
            # regression loss
            loss = nll_loss_regress(Y_batch[:, 1:], mean, logvar, alpha=alpha, mean2=mean2, logvar2=logvar2, F=F, F2=F2, cov_mat=args['cov_mat'], device=device) + regularization
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
                dropout_sample = mc_sample(model, X_val, Y_val, args['n_MC'], device, args['cov_mat'])
                pppp, rmse, mean_norm, logvar_norm = get_scalar_metrics(dropout_sample['mean'], dropout_sample['logvar'], Y_val[:, 1:], args['n_MC'])
                
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
                dropout_result = average_over_dropout(dropout_sample)
                #sampled_result = sample_from_likelihood(dropout_result, n_sample=500)
                #np.save('sample', sampled_result.reshape(Y_val.shape[0], 500*(Y_val.shape[1] - 1)))

                # Convert to natural units
                X_nat, Y_nat, em_nat, em_nat_second = plotting.get_natural_units(X=X_val, Y=Y_val, meta=data_meta, **dropout_result)

                # Get mapping plots
                psFlux_mag = get_magnitude_plot(epoch+1, X_nat.loc[val_sampled[:200], :], Y_nat.loc[val_sampled[:200], :], em_nat.loc[val_sampled[:200], :], em_nat_second.loc[val_sampled[:200], :], 'psFlux_%s', data_meta)
                cModelFlux_mag = get_magnitude_plot(epoch+1, X_nat.loc[val_sampled[:200], :], Y_nat.loc[val_sampled[:200], :], em_nat.loc[val_sampled[:200], :], em_nat_second.loc[val_sampled[:200], :], 'cModelFlux_%s', data_meta)
                psFlux = get_flux_plot(epoch+1, X_nat.loc[val_sampled, :], Y_nat.loc[val_sampled, :], em_nat.loc[val_sampled, :], em_nat_second.loc[val_sampled, :], 'psFlux_%s', data_meta)
                cModelFlux = get_flux_plot(epoch+1, X_nat.loc[val_sampled, :], Y_nat.loc[val_sampled, :], em_nat.loc[val_sampled, :], em_nat_second.loc[val_sampled, :], 'cModelFlux_%s', data_meta)
                moments = get_moment_plot(epoch+1, X_nat.loc[val_sampled, :], Y_nat.loc[val_sampled, :], em_nat.loc[val_sampled, :], em_nat_second.loc[val_sampled, :])
                conf_mat = get_star_metrics(epoch+1, X_nat, Y_nat, em_nat)

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

def average_over_dropout(dropout_sample):
    # FIXME only works for mixture
    dropout_result = OrderedDict(
                    mu = np.mean(dropout_sample['mean'], axis=0),
                    al_sig2 = uncertain.get_aleatoric_sigma2(dropout_sample['logvar']),
                    ep_sig2 = uncertain.get_epistemic_sigma2(dropout_sample['mean']),
                    F = np.mean(dropout_sample['F'], axis=0),

                    mu_second = np.mean(dropout_sample['mean2'], axis=0),
                    al_sig2_second = uncertain.get_aleatoric_sigma2(dropout_sample['logvar2']),
                    ep_sig2_second = uncertain.get_epistemic_sigma2(dropout_sample['mean2']),
                    alpha = np.mean(dropout_sample['alpha'], axis=0),
                    F2 = np.mean(dropout_sample['F2'], axis=0),
                    
                    mu_class = np.mean(sigmoid(dropout_sample['mean_class']), axis=0),
                    al_sig2_class = uncertain.get_aleatoric_sigma2(dropout_sample['logvar_class']),
                    ep_sig2_class = uncertain.get_epistemic_sigma2(dropout_sample['mean_class']),)
    return dropout_result

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sample_from_likelihood(learned_params, n_sample):
    # FIXME only works for mixture
    n_obj, reg_dim = learned_params['mu'].shape
    sample = np.full([n_obj, n_sample, reg_dim], np.nan) # initialize sample tensor
    prob_second = 0.5*sigmoid(learned_params['alpha']).repeat(n_sample, axis=1) # [n_obj, n_sample]
    unif = np.random.rand(n_obj, n_sample)
    second_gaussian = (unif < prob_second)
    first_gaussian = np.logical_not(second_gaussian)
    first_sample = sample_from_lowrank(learned_params['mu'], learned_params['al_sig2'], learned_params['F'], n_sample)
    sample[first_gaussian, :] = first_sample[first_gaussian, :]
    second_sample = sample_from_lowrank(learned_params['mu_second'], learned_params['al_sig2_second'], learned_params['F2'], n_sample)
    sample[second_gaussian, :] = second_sample[second_gaussian, :]
    assert np.isnan(sample).any() == False # entire tensor should be populated
    return sample

def sample_from_lowrank(mu, var, F, n_sample):
    # (24) in Miller et al 2016
    n_obj, reg_dim = mu.shape
    rank = 2 # FIXME
    mu = mu.reshape([n_obj, 1, reg_dim])
    sig = np.sqrt(var).reshape([n_obj, 1, reg_dim])
    F = np.expand_dims(F.reshape(n_obj, reg_dim, rank), axis=1) # [n_obj, 1, reg_dim, rank]
    z_lowrank = np.random.randn(n_obj, n_sample, 1, rank)
    z_diag = np.random.randn(n_obj, n_sample, reg_dim)
    
    x = np.sum(F*z_lowrank, axis=3) # [n_obj, n_sample, reg_dim]
    x += mu
    x += sig * z_diag
    return x

def l2_norm(pred):
    norm_per_data = np.linalg.norm(pred, axis=2) # shape [n_MC, n_data]
    return np.mean(norm_per_data)

def nll_loss_regress(true, mean, logvar, device, F=None, mean2=None, logvar2=None, F2=None, alpha=None, cov_mat='low_rank'):
    if cov_mat == 'diagonal':
        nll_loss_diagonal(true, mean, logvar)
    elif cov_mat == 'low_rank':
        return nll_loss_lowrank(true, mean, logvar, device, F)
    elif cov_mat == 'mixture':
        batch_size, _ = mean.shape
        rank = 2 #FIXME
        log_nll = torch.empty([batch_size, rank], device=device)
        logsigmoid = torch.nn.LogSigmoid()
        # FIXME rank hardcode
        alpha = alpha.reshape(-1)
        log_nll[:, 0] = torch.log(torch.tensor([0.5], device=device)) + logsigmoid(-alpha) + nll_loss_lowrank(true, mean, logvar, device=device, F=F, reduce=False) # [batch_size]
        log_nll[:, 1] = torch.log(torch.tensor([0.5], device=device)) + logsigmoid(alpha) + nll_loss_lowrank(true, mean2, logvar2, device=device, F=F2, reduce=False) # [batch_size]
        sum_two_gaus = torch.logsumexp(log_nll, dim=1) 
        return torch.mean(sum_two_gaus)

def nll_loss_diagonal(true, mean, logvar):
    precision = torch.exp(-logvar)
    return torch.mean(torch.sum(precision * (true - mean)**2.0 + logvar, dim=1), dim=0)

def nll_loss_lowrank(true, mean, logvar, device, F=None, reduce=True):
    # 1/(Y_dim - 1) * (sq_mahalanobis + log(det of \Sigma))
    batch_size, reg_dim = mean.shape # reg_dim = Y_dim - 1
    rank = 2
    F = F.reshape([batch_size, reg_dim, rank]) # FIXME: hardcoded for rank 2
    inv_var = torch.exp(-logvar) # [batch_size, reg_dim]
    diag_inv_var = torch.diag_embed(inv_var)  # [batch_size, reg_dim, reg_dim]
    diag_prod = F**2.0 * inv_var.reshape([batch_size, reg_dim, 1]) # [batch_size, reg_dim, rank] after broadcasting
    off_diag_prod = torch.prod(F, dim=2)*inv_var # [batch_size, reg_dim]
    #batchdiag = torch.diag_embed(torch.exp(logvar)) # [batch_size, reg_dim, reg_dim]
    #batch_eye = torch.eye(rank).reshape(1, rank, rank).repeat(batch_size, 1, 1) # [batch_size, rank, rank]
    #assert batchdiag.shape == torch.Size([batch_size, reg_dim, reg_dim])

    # (25), (26) in Miller et al 2016
    log_det = torch.sum(logvar, dim=1) # [batch_size]
    M00 = torch.sum(diag_prod[:, :, 0], dim=1) + 1.0 # [batch_size]
    M11 = torch.sum(diag_prod[:, :, 1], dim=1) + 1.0 # [batch_size]
    M12 = torch.sum(off_diag_prod, dim=1) # [batch_size]
    assert M00.shape == torch.Size([batch_size])
    assert M12.shape == torch.Size([batch_size])
    det_M = M00*M11 - M12**2.0 # [batch_size]
    assert det_M.shape == torch.Size([batch_size])
    assert log_det.shape == torch.Size([batch_size])
    log_det += torch.log(det_M) 
    assert log_det.shape == torch.Size([batch_size])
    #print(det_M)

    inv_M = torch.ones([batch_size, rank, rank], device=device)
    inv_M[:, 0, 0] = M11
    inv_M[:, 1, 1] = M00
    inv_M[:, 1, 0] = -M12
    inv_M[:, 0, 1] = -M12
    inv_M /= det_M.reshape(batch_size, 1, 1)

    # (27) in Miller et al 2016
    inv_cov = diag_inv_var - torch.bmm(torch.bmm(torch.bmm(torch.bmm(diag_inv_var, F), inv_M), torch.transpose(F, 1, 2)), diag_inv_var) 
    assert inv_cov.shape == torch.Size([batch_size, reg_dim, reg_dim])
    sq_mahalanobis = torch.squeeze(torch.bmm(torch.bmm((mean - true).reshape(batch_size, 1, reg_dim), inv_cov), (mean - true).reshape(batch_size, reg_dim, 1)))
    assert sq_mahalanobis.shape == torch.Size([batch_size])

    if reduce==True:
        return torch.mean(sq_mahalanobis + log_det, dim=0)
    else:
        return sq_mahalanobis + log_det

def nll_loss_classify(true, mean, logvar):
    #precision = torch.exp(-logvar)
    #return torch.mean(torch.sum(precision * (true - mean)**2.0 + logvar, dim=1), dim=0)
    loss = torch.nn.BCEWithLogitsLoss()
    return loss(mean, true)

def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

def mc_sample(model, X_val, Y_val, n_MC, device, cov_mat):
    n_val, Y_dim = Y_val.shape
    rank = 2 # FIXME
    MC_samples = [model(Variable(torch.FloatTensor(X_val)).to(device)) for _ in range(n_MC)] # shape [K, N, 2D]
    # FIXME: very inefficient tuple of tuples...
    dropout_sample = OrderedDict(mean = torch.stack([tup[0] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy(),
                            logvar = torch.stack([tup[1] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy(),
                            mean_class = torch.stack([tup[7] for tup in MC_samples]).view(n_MC, n_val, 1).cpu().data.numpy(),
                            logvar_class = torch.stack([tup[8] for tup in MC_samples]).view(n_MC, n_val, 1).cpu().data.numpy(),
                            F = None,
                            mean2 = None,
                            logvar2 = None,
                            F2 = None,
                            alpha = None,)
    if cov_mat == 'diagonal':
        return dropout_sample
    elif cov_mat=='low_rank':
        dropout_sample['F'] = torch.stack([tup[2] for tup in MC_samples]).view(n_MC, n_val, (Y_dim - 1)*rank).cpu().data.numpy()
        return dropout_sample
    elif cov_mat == 'mixture':
        dropout_sample['F'] = torch.stack([tup[2] for tup in MC_samples]).view(n_MC, n_val, (Y_dim - 1)*rank).cpu().data.numpy()
        dropout_sample['mean2'] = torch.stack([tup[3] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy()
        dropout_sample['logvar2'] = torch.stack([tup[4] for tup in MC_samples]).view(n_MC, n_val, Y_dim - 1).cpu().data.numpy()
        dropout_sample['F2'] = torch.stack([tup[5] for tup in MC_samples]).view(n_MC, n_val, (Y_dim - 1)*rank).cpu().data.numpy()
        dropout_sample['alpha'] = torch.stack([tup[6] for tup in MC_samples]).view(n_MC, n_val, 1).cpu().data.numpy()
        return dropout_sample

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

def get_star_metrics(epoch, X, Y, emulated):
    my_dpi = 72.0
    fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
    canvas = plotting.plot_confusion_matrix(fig, X, Y, emulated)
    width, height = fig.get_size_inches() * fig.get_dpi()
    conf_mat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
    return conf_mat

def get_moment_plot(epoch, X, Y, emulated, emulated_second):
    my_dpi = 72.0
    per_filter = []
    for moment_type in ['Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF', 'ra_offset', 'dec_offset']:
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_moment(fig, X, Y, emulated, emulated_second, moment_type)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def get_flux_plot(epoch, X, Y, emulated, emulated_second, flux_formatting, data_meta):
    my_dpi = 72.0
    per_filter = []
    for bp in 'ugrizy':
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_flux(fig, X, Y, emulated, emulated_second, flux_formatting, bp)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def get_magnitude_plot(epoch, X, Y, emulated, emulated_second, flux_formatting, data_meta):
    my_dpi = 72.0
    per_filter = []
    for bp in 'ugrizy':
        fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
        canvas = plotting.plot_magnitude(fig, X, Y, emulated, emulated_second, flux_formatting, bp)
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
        per_filter.append(img)
    all_filters = np.concatenate(per_filter, axis=0)
    #np.save('img_%d' %epoch, img)
    return all_filters

def get_sample_cornerplot(Y_nat, sampled_result):
    n_obj, n_sample, reg_dim = sampled_result.shape
    my_dpi = 72.0
    fig = Figure(figsize=(720/my_dpi, 360/my_dpi), dpi=my_dpi, tight_layout=True)
    canvas = plotting.plot_sample_corner(fig, X, Y, emulated, flux_formatting, bp)
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(1, int(height), int(width), 3)
    return img

if __name__ == '__main__':
    pass