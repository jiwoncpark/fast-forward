import pandas as pd
import numpy as np
import units_utils as units
import sqlite3

def read_opsim_db(opsim_db_path='/global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db'):
    """
    Returns
    -------
    DataFrames of ObsHistory and Field tables in the OpSim DB
    """
    conn = sqlite3.connect(opsim_db_path)
    # Check the table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    print(table_names)
    # Turn table into Pandas df
    obs_history = pd.read_sql(sql="SELECT * from ObsHistory", con=conn, index_col=None)
    field = pd.read_sql(sql="SELECT * from Field", con=conn, index_col=None)
    return obs_history, field

def format_opsim(obs_history, field, save_path=None):
    """
    Parameters
    ----------
    obs_history : Pandas.DataFrame
    field : Pandas.DataFrame
    save_to_disk : str
    
    Note
    ----
    We use the dithered RA, Dec and express all positions in arcsec.
    
    Returns
    -------
    DataFrame obs_history, formatted with new column conventions and units
    """
    # Join with Field table
    obs_history = pd.merge(obs_history, field, left_on='Field_fieldID', right_on='fieldID')
    # Some unit conversion and column renaming
    # NOTE: OpSim DB defines dithered positions as offset from the field center.
    obs_history['ditheredRA'] = obs_history['ditheredRA'].values + obs_history['fieldRA'].values
    obs_history['ditheredDec'] = obs_history['ditheredDec'].values  + obs_history['fieldDec'].values
    obs_history['ditheredRA_asec'] = units.deg_to_arcsec(obs_history['ditheredRA'].values)
    obs_history['ditheredDec_asec'] = units.deg_to_arcsec(obs_history['ditheredDec'].values)
    obs_history['PSF_sigma2'] = units.fwhm_to_sigma(obs_history['finSeeing'].values)**2.0
    obs_history['m5_flux'] = units.mag_to_flux(obs_history['fiveSigmaDepth'].values-22.5)/5.0 
    obs_history['filtSkyBrightness_flux'] = units.mag_to_flux(obs_history['filtSkyBrightness'].values, to_unit='nMgy')
    # Only keep columns we'll need
    #obs_keep_cols = ['obsHistID', 'fieldID', 'expMJD', 'fiveSigmaDepth', 'filtSkyBrightness', 
    #                 'ditheredRA', 'ditheredDec', 'ditheredRA_asec', 'ditheredDec_asec',
    #                 'PSF_sigma2', 'm5_flux', 'filter', 'airmass']
    #obs_history = obs_history[obs_keep_cols]
    if save_path is not None:
        obs_history.to_csv(save_path, index=False)
    return obs_history

def join_src_opsim(src_df, opsim_df, save_path=None):
    """
    Parameters
    ----------
    src_df : Pandas.DataFrame
    opsim_df : Pandas.DataFrame
    save_to_disk : str
    
    Note
    ----
    In OpSim, obsHistID and filter combinations are unique, i.e.
    >> opsim.groupby(['obsHistID', 'filter']).size().prod() == 1
    The tracts mapping table is not used.
    
    Returns
    -------
    DataFrame of Source catalog joined with the OpSim DB and a dictionary
    containing column names from OpSim and Source catalogs
    """
    opsim_cols = opsim_df.columns.values
    src_cols = src_df.columns.values
    metadata = {'opsim_cols': opsim_cols, 'src_cols': src_cols}
    joined = src_df.merge(opsim_df, how='inner', left_on=['visit', 'filter'], right_on=['obsHistID', 'filter'], validate='m:1')
    joined['n_obs'] = joined['objectId'].map(joined['objectId'].value_counts()) # Number of visits for each object
    if save_path is not None:
        joined.to_csv(save_path, index=False)
    return joined, metadata

def join_obj_opsim(obj_df, src_opsim_df, save_path=None):
    """
    Parameters
    ----------
    obj_df : Pandas.DataFrame
    src_opsim_df : Pandas.DataFrame
    save_to_disk : str
    
    Note
    ----
    src_opsim_df must contain columns 'objectId' and opsim_avg_cols below
    (the first return value of join_src_opsim)
    
    Returns
    -------
    DataFrame of Source catalog joined with the OpSim DB and a dictionary
    containing column names from OpSim and Source catalogs
    """
    opsim_avg_cols = ['m5_flux', 'PSF_sigma2', 'filtSkyBrightness_flux', 'airmass', 'dist2Moon', 'n_obs']
    avg_opsim = src_opsim_df.groupby(['objectId'], as_index=False)[opsim_avg_cols].mean()
    
    obj_cols = obj_df.columns.values
    metadata = {'opsim_cols': opsim_avg_cols, 'obj_cols': obj_cols}
    
    joined = obj_df.merge(avg_opsim, how='inner', left_on='objectId', right_on='objectId', validate='1:m')
    if save_path is not None:
        joined.to_csv(save_path, index=False)
    return joined, metadata

def read_tracts_mapping():
    tracts_mapping_path = '/global/cscratch1/sd/desc/DC2/data/Run1.2i/rerun/281118/tracts_mapping.sqlite3'
    conn = sqlite3.connect(tracts_mapping_path)
    # Check the table name
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_name = cursor.fetchall()[0][0]
    print("Table name: ", table_name)
    # Turn table into Pandas df
    overlaps = pd.read_sql(sql="SELECT * from '%s'" %table_name, con=conn)
    return overlaps