import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import units_utils as units

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
        
        if 'star' not in self.X_base_cols:
            self.X_base_cols.append('star')
            
        # Initialize new column mapping and names
        self.X_col_map = dict(zip(self.X_base_cols, self.X_base_cols)) # same by default
        self.Y_col_map = dict(zip(self.Y_base_cols, self.Y_base_cols))
        
        XY = pd.read_csv(data_path, index_col=None)
        self.X = XY[self.X_base_cols]
        self.Y = XY[self.Y_base_cols]
        
        if self.ignore_null_rows:
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
        self.X_cat_cols = [self.X_col_map['star']]
        self.X_cat_mapping = dict(zip(self.X_cols, range(len(self.X_cols))))
        
        # Normalize features
        self.trainval_indices, self.train_indices, self.val_indices, self.n_train = self.split_train_val()
        self.normalize_XY(exclude_X_cols=self.X_cat_cols, normalize_Y=True)
        
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
    
    def export_metadata_for_eval(self):
        import json
        
        eval_metadata = {
                        'X_cat_cols': self.X_cat_cols,
                        'X_cols': self.X_cols,
                        'Y_cols': self.Y_cols,
                        'val_indices': self.val_indices.tolist(),
                        'X_mean': self.X_mean.tolist(),
                        'Y_mean': self.Y_mean.tolist(),
                        'X_std': self.X_std.tolist(),
                        'Y_std': self.Y_std.tolist(),}
        
        with open('eval_metadata.txt', 'w') as fp:
            json.dump(eval_metadata, fp)
    
    def engineer_XY(self):
        """Engineer features in input X and label Y
        
        Note
        ----
        Does not include normalization
        """
        
        # Negate star feature to approximate "extendedness"
        self.X['not_star'] = 1.0 - self.X['star']
        self.X_col_map['star'] = 'not_star'
        
        # Turn fluxes into magnitudes
        for mag_name in ['u', 'g', 'r', 'i', 'z', 'y_truth']: #FIXME: suffixing for y
            mag = self.X[mag_name].values
            flux = units.mag_to_flux(mag, to_unit='nMgy')
            flux_name = mag_name + '_flux'
            self.X[flux_name] = flux
            self.X_col_map[mag_name] = flux_name

        # Calculate positional offset
        if 'ra_obs' in self.Y_base_cols:
            assert 'ra_truth' in self.X_base_cols
            self.Y['ra_offset'] =  self.Y['ra_obs'] - self.X['ra_truth'] 
            self.Y_col_map['ra_obs'] = 'ra_offset'
        if 'dec_obs' in self.Y_base_cols:
            assert 'dec_truth' in self.X_base_cols
            self.Y['dec_offset'] = self.Y['dec_obs'] - self.X['dec_truth']
            self.Y_col_map['dec_obs'] = 'dec_offset'
            
        # Square root the second moments
        if 'Ixx' in self.Y_base_cols:
            self.Y['Ixx'] = np.sqrt(self.Y['Ixx'])
        if 'Iyy' in self.Y_base_cols:
            self.Y['Iyy'] = np.sqrt(self.Y['Iyy'])
        
        # Get first moments in asec
        if 'x' in self.Y_base_cols:
            self.Y['x'] = self.Y['x']/self.pixel_scale
            self.Y['y_obs'] = self.Y['y_obs']/self.pixel_scale
        
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
        self.X.loc[(self.X['star']==False) & (self.X['bulge_to_total_ratio_i']==0.0), ['ellipticity_1_bulge_true', 'ellipticity_2_bulge_true']] = 0.0
    
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
        
    def split_train_val(self):
        n_train = int(self.n_trainval*self.train_frac)
        trainval_indices = np.arange(self.n_trainval)
        np.random.shuffle(trainval_indices)
        train_indices, val_indices = trainval_indices[:n_train], trainval_indices[n_train:]
        return trainval_indices, train_indices, val_indices, n_train
    
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
        X_mean[exclude_X_cols] = 0.0
        X_std[exclude_X_cols] = 1.0
        self.X_mean = X_mean
        self.X_std = X_std
        self.X.iloc[self.train_indices, :] = (X_train - self.X_mean)/self.X_std
        self.X.iloc[self.val_indices, :] = (X_val - self.X_mean)/self.X_std
        
        if normalize_Y:
            Y_train = self.Y.iloc[self.train_indices, :].copy()
            Y_val = self.Y.iloc[self.val_indices, :].copy()
            Y_mean = Y_train.mean()
            Y_std = Y_train.std()
            Y_mean[exclude_Y_cols] = 0.0
            Y_std[exclude_Y_cols] = 1.0
            self.Y_mean = Y_mean
            self.Y_std = Y_std
            self.Y.loc[self.train_indices, :] = (Y_train - self.Y_mean)/self.Y_std
            self.Y.loc[self.val_indices, :] = (Y_val - self.Y_mean)/self.Y_std

        if self.verbose:
            print("Standardized X except: ", exclude_X_cols)
            print("Standardized Y except: ", exclude_Y_cols)
        #    print("Normalizing columns in X: ", self.X_cols)
        #    print("Normalizing columns in Y: ", self.Y_cols)
        
        
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    
    # Test constructor
    data = DerpData(data_path='obj_master.csv', X_base_cols=['star', 'ra_truth'], Y_base_cols=['ra_obs'],
                        ignore_null_rows=True, verbose=True)
    
    # Test __getitem__
    X_slice, Y_slice = data[0]
    print(X_slice.shape, Y_slice.shape)
    
    # Test loader instantiated with DerpData instance
    train_sampler = SubsetRandomSampler(data.train_indices)
    train_loader = DataLoader(data, batch_size=7, sampler=train_sampler)
    for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
        print(X_batch.shape)
        print(Y_batch.shape)
        break