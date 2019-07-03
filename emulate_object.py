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
truth_cols = list('ugriz') + ['y_truth', 'ra_truth', 'dec_truth', 'redshift',] #'star',]
truth_cols += ['size_bulge_true', 'size_minor_bulge_true', 'ellipticity_1_bulge_true', 'ellipticity_2_bulge_true', 'bulge_to_total_ratio_i']
truth_cols += ['size_disk_true', 'size_minor_disk_true', 'ellipticity_1_disk_true', 'ellipticity_2_disk_true',]
opsim_cols = ['m5_flux', 'PSF_sigma2', 'filtSkyBrightness_flux', 'airmass', 'n_obs']
# Y base columns
drp_cols = ['x', 'y_obs', 'ra_obs', 'dec_obs', 'Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF',] #'extendedness',]
drp_cols_prefix = ['cModelFlux_', 'psFlux_']
drp_cols_suffix = ['_base_CircularApertureFlux_70_0_instFlux','_ext_photometryKron_KronFlux_instFlux',]
drp_cols += [t[0] + t[1] for t in list(itertools.product(drp_cols_prefix, list('ugrizy')))]
drp_cols += [t[1] + t[0] for t in list(itertools.product(drp_cols_suffix, list('ugrizy')))]

# Define dataset
data = DerpData(data_path='raw_data/brighter_gals_obj_master.csv', X_base_cols=truth_cols + opsim_cols, Y_base_cols=drp_cols, 
                verbose=args['verbose'], ignore_null_rows=True, save_to_disk=True)
X_cols = data.X_cols
Y_cols = data.Y_cols
n_trainval = data.n_trainval
n_train = data.n_train
n_val = n_trainval - n_train
X_dim = data.X_dim
Y_dim = data.Y_dim

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

# Split train vs. val
train_sampler = SubsetRandomSampler(data.train_indices)
val_sampler = SubsetRandomSampler(data.val_indices)

# Define dataloader
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
train_loader = DataLoader(data, batch_size=args['batch_size'], sampler=train_sampler, **kwargs)
val_loader = DataLoader(data, batch_size=args['batch_size'], sampler=val_sampler, **kwargs)

# Test dataloader
for batch_idx, (X_batch, Y_batch) in enumerate(val_loader):
    print(X_batch.shape)
    print(Y_batch.shape)
    break

#####################
# Training metadata #
#####################

data.export_metadata_for_eval(device_type=device.type)

#########
# Model #
#########

from models import ConcreteDense

length_scale = args['l']
wr = length_scale**2.0/data.n_train
dr = 2.0/data.n_train
model = ConcreteDense(data.X_dim, data.Y_dim, args['n_features'], wr, dr).to(device)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#############
# Optimizer #
#############

from solver import fit_model
from torch import optim

X_val = data.X[data.val_indices, :]
Y_val = data.Y[data.val_indices, :]
optimizer = optim.Adam(params=model.parameters(), lr=5e-4, amsgrad=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 800, 1600], gamma=0.1)
checkpoint_path = None if args['checkpoint_path'] is False else args['checkpoint_path']

model = fit_model(model, optimizer, lr_scheduler, args['n_epochs'], train_loader, val_loader,
                  device=device, logging_interval=args['logging_interval'], checkpointing_interval=args['checkpointing_interval'],
                  X_val=X_val, Y_val=Y_val, n_MC=args['n_MC'], run_id=args['run_id'], checkpoint_path=checkpoint_path)

