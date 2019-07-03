import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import units_utils as units # FIXME: may not be used
import astropy.units as u

class DerpData(Dataset):
    """Preprocessed and unnormalized Derp dataset."""

    def __init__(self, data_path, X_base_cols, Y_base_cols, train_frac=0.9, ignore_null_rows=True, verbose=True, save_to_disk=False):
        """
        Parameters
        ----------
        truth_path : string
            Path to csv file containing the input X
        drp_path : string
            Path to csv file containing the label Y
        ignore_null_rows : Bool
            Whether rows with null values will be ignored
        """
        self.verbose = verbose
        self.save_to_disk = save_to_disk
        #self.mask_val = -1
        self.ignore_null_rows = ignore_null_rows
        self.X_base_cols = X_base_cols
        self.Y_base_cols = Y_base_cols
        self.star_cols = list('ugriz') + ['y_truth', 'ra_truth', 'dec_truth', 'redshift', 'star', 'agn'] 
        self.pixel_scale = 0.2 # arcsec/pix
        self.train_frac = train_frac
        
        #if 'truth_id' not in self.X_base_cols + self.Y_base_cols:
        #    self.Y_base_cols.append('truth_id')
            
        # Initialize new column mapping and names
        self.X_col_map = dict(zip(self.X_base_cols, self.X_base_cols)) # same by default
        self.Y_col_map = dict(zip(self.Y_base_cols, self.Y_base_cols))
        
        XY = pd.read_csv(data_path, index_col=None)
        self.X = XY[self.X_base_cols]
        self.Y = XY[self.Y_base_cols]

        if self.ignore_null_rows:
            if 'star' in self.X_base_cols:
                self.zero_extragal_cols_for_stars()
            self.zero_nan_for_galaxies()
            self.delete_null_rows()
        else:
            raise NotImplementedError
            #self.mask_null_values()

        # Engineer features
        self.engineer_XY()

        # Slice features
        self.X_cols =  list(self.X_col_map.values())
        self.Y_cols = list(self.Y_col_map.values())
        self.X = self.X[self.X_cols]
        self.Y = self.Y[self.Y_cols]
        
        # Save metadata: number of examples, input dim, output dim
        self.n_trainval, self.X_dim = self.X.shape
        _, self.Y_dim = self.Y.shape
        
        # Categorical data
        if 'star' in self.X_base_cols:
            self.X_cat_cols = [self.X_col_map['star'],]
        else:
            self.X_cat_cols = []
        
        # Split train vs. val
        #self.val_indices = np.load('val_indices.npy')
        self.val_indices = np.arange(int((1.0 - train_frac)*self.n_trainval))
        self.train_indices = np.array(list(set(range(self.n_trainval)) - set(self.val_indices)))
        self.n_val = len(self.val_indices)
        self.n_train = len(self.train_indices)

        # Normalize features
        self.normalize_XY(exclude_X_cols=self.X_cat_cols + [], normalize_Y=False)

        # Save processed data to disk
        if self.save_to_disk:
            if self.verbose:
                print("Saving processed data to disk...")
            self.X.to_csv('data/processed_X.csv', index=False)
            self.Y.to_csv('data/processed_Y.csv', index=False)

        # Convert into numpy array
        self.X = self.X.values.astype(np.float32)
        self.Y = self.Y.values.astype(np.float32)
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample_X = self.X[idx, :]
        sample_Y  = self.Y[idx, :]
        return sample_X, sample_Y
    
    def export_metadata_for_eval(self, device_type):
        import json
        
        data_meta = {
                    'device_type': device_type,
                    'X_dim': self.X_dim,
                    'Y_dim': self.Y_dim,
                    'n_val': self.n_val,
                    'n_train': self.n_train,
                    'X_cat_cols': self.X_cat_cols,
                    'X_cols': self.X_cols,
                    'Y_cols': self.Y_cols, 
                    'train_indices': self.train_indices.tolist(),                       
                    'val_indices': self.val_indices.tolist(),
                    'X_mean': self.X_mean.tolist(),
                    'Y_mean': self.Y_mean.tolist(),
                    'X_std': self.X_std.tolist(),
                    'Y_std': self.Y_std.tolist(),}
        
        with open('data_meta.txt', 'w') as fp:
            json.dump(data_meta, fp)
    
    def engineer_XY(self):
        """Engineer features in input X and label Y
        
        Note
        ----
        Does not include normalization
        """
        
        # Negate star feature to approximate "extendedness"
        if 'star' in self.X_base_cols:
            self.X['not_star'] = 1.0 - self.X['star']
            self.X_col_map['star'] = 'not_star'
        
        # Turn fluxes into magnitudes
        for mag_name in ['u', 'g', 'r', 'i', 'z', 'y_truth']: #FIXME: suffixing for y
            mag = self.X[mag_name].values
            flux = (mag * u.ABmag).to_value(u.Jy)
            flux_name = mag_name + '_flux'
            self.X[flux_name] = flux
            self.X_col_map[mag_name] = flux_name

        # Calculate positional offset
        if 'ra_obs' in self.Y_base_cols:
            assert 'ra_truth' in self.X_base_cols
            self.Y['ra_offset'] =  (self.Y['ra_obs'] - self.X['ra_truth'])*3600.0 # asec
            self.Y_col_map['ra_obs'] = 'ra_offset'
        if 'dec_obs' in self.Y_base_cols:
            assert 'dec_truth' in self.X_base_cols
            self.Y['dec_offset'] = (self.Y['dec_obs'] - self.X['dec_truth'])*3600.0 # asec
            self.Y_col_map['dec_obs'] = 'dec_offset'
            
        # Square root the second moments
        if 'Ixx' in self.Y_base_cols:
            self.Y.loc[:, 'Ixx'] = np.sqrt(self.Y['Ixx'])/3600.0 # asec --> deg
            self.Y.loc[:, 'IxxPSF'] = np.sqrt(self.Y['IxxPSF'])
        if 'Iyy' in self.Y_base_cols:
            self.Y.loc[:, 'Iyy'] = np.sqrt(self.Y['Iyy'])/3600.0 # asec --> deg
            self.Y.loc[:, 'IyyPSF'] = np.sqrt(self.Y['IyyPSF'])
        if 'Ixy' in self.Y_base_cols:
            self.Y.loc[:, 'Ixy'] = self.Y['Ixy']/1000.0 # 1000 asec^2
        
        # Get first moments in asec
        if 'x' in self.Y_base_cols:
            ref_centroid = 350000.0 # arcsec
            self.Y.loc[:, 'x'] = (self.Y['x']/self.pixel_scale - ref_centroid)/3600.0/1000.0 # asec --> deg --> 1000 deg
            self.Y.loc[:, 'y_obs'] = (self.Y['y_obs']/self.pixel_scale - ref_centroid)/3600.0/1000.0 # asec --> deg --> 1000 deg
        
        for col in self.Y_base_cols:
            if 'Flux' in col:
                self.Y.loc[:, col] = self.Y.loc[:, col]*1.e-9*1000.0 # microJy

    def zero_extragal_cols_for_stars(self):
        """Zeroes out the extragal columns for stars
        """
        extragal_cols = list(set(self.X_base_cols) - set(self.star_cols))
        self.X.loc[self.X['star']==True, extragal_cols] = 0.0
        
    def zero_nan_for_galaxies(self):
        """Zeroes out some extragal columns for galaxies
        
        Note
        ----
        Galaxies with bulge ratio = 0.0 have NaNs as ellipticities 1 and 2
        """
        self.X.loc[(self.X['bulge_to_total_ratio_i']==0.0), ['ellipticity_1_bulge_true', 'ellipticity_2_bulge_true']] = 0.0
    
    def delete_null_rows(self):
        """Deletes rows with any null value
        Note
        ----
        This method assumes self.X has no null value.
        
        """
        n_rows_before = len(self.X)
        y_notnull_rows = np.logical_not(self.Y.isna().any(1))
        self.X = self.X.loc[y_notnull_rows, :]
        self.Y = self.Y.loc[y_notnull_rows, :]
        n_rows_after = len(self.X)
        self.X.reset_index(drop=True, inplace=True)
        self.Y.reset_index(drop=True, inplace=True)
        if self.verbose:
            print("Deleting null rows: %d --> %d" %(n_rows_before, n_rows_after))
        
    def mask_null_values(self):
        """Replaces null values with a token, self.mask_val
        
        """
        self.X.fillna(self.mask_val, inplace=True)
        self.y.fillna(self.mask_val, inplace=True)
    
    def normalize_XY(self, exclude_X_cols=[], exclude_Y_cols=[], normalize_Y=True):
        """Standardizes input X and label Y column-wise except exclude_cols
        
        Note
        ----
        The validation set must be standardized using the parameters of the training set.
        """
        X_train = self.X.iloc[self.train_indices, :].copy()
        X_val = self.X.iloc[self.val_indices, :].copy()
        X_mean = X_train.mean()
        X_std = X_train.std()
        X_mean.loc[exclude_X_cols] = 0.0
        X_std.loc[exclude_X_cols] = 1.0
        self.X_mean = X_mean
        self.X_std = X_std
        self.X.iloc[self.train_indices, :] = (X_train - self.X_mean)/self.X_std
        self.X.iloc[self.val_indices, :] = (X_val - self.X_mean)/self.X_std
        
        if normalize_Y:
            Y_train = self.Y.iloc[self.train_indices, :].copy()
            Y_val = self.Y.iloc[self.val_indices, :].copy()
            Y_mean = Y_train.mean()
            Y_std = Y_train.std()
            Y_mean.loc[exclude_Y_cols] = 0.0
            Y_std.loc[exclude_Y_cols] = 1.0
            self.Y_mean = Y_mean
            self.Y_std = Y_std
            self.Y.iloc[self.train_indices, :] = (Y_train - self.Y_mean)/self.Y_std
            self.Y.iloc[self.val_indices, :] = (Y_val - self.Y_mean)/self.Y_std
        else:
            self.Y_mean = np.zeros((self.Y_dim,))
            self.Y_std = np.ones((self.Y_dim,))

        if self.verbose:
            print("Standardized X except: ", exclude_X_cols)
            print("Standardized Y except: ", exclude_Y_cols)
        #    print("Normalizing columns in X: ", self.X_cols)
        #    print("Normalizing columns in Y: ", self.Y_cols)
        
        
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    
    import itertools

    # X base columns
    truth_cols = list('ugriz') + ['y_truth', 'ra_truth', 'dec_truth', 'redshift', 'star',]
    truth_cols += ['size_bulge_true', 'size_minor_bulge_true', 'ellipticity_1_bulge_true', 'ellipticity_2_bulge_true', 'bulge_to_total_ratio_i']
    truth_cols += ['size_disk_true', 'size_minor_disk_true', 'ellipticity_1_disk_true', 'ellipticity_2_disk_true',]
    opsim_cols = ['m5_flux', 'PSF_sigma2', 'filtSkyBrightness_flux', 'airmass', 'n_obs']
    # Y base columns
    drp_cols = ['x', 'y_obs', 'ra_obs', 'dec_obs', 'Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF',] #'extendedness',]
    drp_cols_prefix = ['cModelFlux_', 'psFlux_']
    drp_cols_suffix = ['_base_CircularApertureFlux_70_0_instFlux','_ext_photometryKron_KronFlux_instFlux',]
    drp_cols += [t[0] + t[1] for t in list(itertools.product(drp_cols_prefix, list('ugrizy')))]
    drp_cols += [t[1] + t[0] for t in list(itertools.product(drp_cols_suffix, list('ugrizy')))]

    # Test constructor
    data = DerpData(data_path='raw_data/obj_master.csv', X_base_cols=truth_cols+opsim_cols, Y_base_cols=drp_cols,
                        ignore_null_rows=True, verbose=True, save_to_disk=False)
    data.export_metadata_for_eval(device_type='cuda')

    # Test __getitem__
    X_slice, Y_slice = data[0]
    print(X_slice.shape, Y_slice.shape)
    
    # Test loader instantiated with DerpData instance
    train_sampler = SubsetRandomSampler(data.train_indices)
    train_loader = DataLoader(data, batch_size=7, sampler=train_sampler)
    print(train_loader.dataset.n_train, train_loader.dataset.n_val)
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        print(X_batch.shape)
        print(Y_batch.shape)
        break