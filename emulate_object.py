import torch
import numpy as np

#################
# Global config #
#################

import json

np.random.seed(2809)
torch.manual_seed(2809)
torch.cuda.manual_seed(2809)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device=='cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
print("device: ", device)
    
args = json.load(open("args.txt"))

############
# Data I/O #
############

from derp_data import DerpData
import itertools

# X base columns
truth_cols = list('ugrizy') + ['ra_truth', 'dec_truth', 'redshift', 'star',]
truth_cols += ['mag_true_%s_lsst' %bp for bp in 'ugrizy']
truth_cols += ['size_bulge_true', 'size_minor_bulge_true', 'ellipticity_1_bulge_true', 'ellipticity_2_bulge_true', 'bulge_to_total_ratio_i']
truth_cols += ['size_disk_true', 'size_minor_disk_true', 'ellipticity_1_disk_true', 'ellipticity_2_disk_true',]
opsim_cols = ['m5_flux', 'PSF_sigma2', 'filtSkyBrightness_flux', 'airmass', 'n_obs']
# Y base columns
drp_cols = ['Ix', 'Iy', 'ra_obs', 'dec_obs', 'Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF',] #'extendedness',]
drp_cols_prefix = ['cModelFlux_', 'psFlux_']
drp_cols_suffix = []
#drp_cols_suffix = ['_ext_photometryKron_KronFlux_instFlux', '_base_CircularApertureFlux_70_0_instFlux', 
drp_cols += [t[0] + t[1] for t in list(itertools.product(drp_cols_prefix, list('ugrizy')))]
drp_cols += [t[1] + t[0] for t in list(itertools.product(drp_cols_suffix, list('ugrizy')))]


# Define dataset
data = DerpData(data_path='raw_data/obj_master_tract4850.csv', X_base_cols=truth_cols + opsim_cols, Y_base_cols=drp_cols, 
                args=args, ignore_null_rows=True, save_to_disk=True)
if not args['data_already_processed']:
    data.export_metadata_for_eval(device_type=device.type)
# Read metadata if reading processed data from disk:
data_meta = json.load(open("data_meta.txt"))

X_cols = data_meta['X_cols']
Y_cols = data_meta['Y_cols']
train_indices = data_meta['train_indices']
val_indices = data_meta['val_indices']
X_dim = data_meta['X_dim']
Y_dim = data_meta['Y_dim']

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# Split train vs. val
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Define dataloader
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
train_loader = DataLoader(data, batch_size=args['batch_size'], sampler=train_sampler, **kwargs)
val_loader = DataLoader(data, batch_size=args['batch_size'], sampler=val_sampler, **kwargs)

# Test dataloader
for batch_idx, (X_batch, Y_batch) in enumerate(val_loader):
    print(X_batch.shape)
    print(Y_batch.shape)
    break

#########
# Model #
#########

from models import ConcreteDense

length_scale = args['l']
n_train = len(train_indices)
wr = length_scale**2.0/n_train
dr = 2.0/n_train
model = ConcreteDense(X_dim, Y_dim, args['n_features'], wr, dr).to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#############
# Optimizer #
#############

from solver import fit_model
from torch import optim

X_val = data.X[val_indices, :]
Y_val = data.Y[val_indices, :]
optimizer = optim.Adam(params=model.parameters(), lr=args['lr'], amsgrad=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 800, 1300, 1800, 2300, 2600, 2800, 3100, 3300, 3600, 3800, 4100, 4200, 4300, 4400], gamma=0.7)
checkpoint_path = None if args['checkpoint_path'] is False else args['checkpoint_path']

model = fit_model(model, optimizer, lr_scheduler, train_loader, val_loader,
                  device=device, args=args, data_meta=data_meta,
                  X_val=X_val, Y_val=Y_val, checkpoint_path=checkpoint_path)

