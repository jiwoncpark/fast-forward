import os, sys
import numpy as np
import astropy.units as u
sys.path.insert(0, '/home/jwp/stage/sl/LSSTDESC/sims_photUtils/python') # not necessary in desc-stack kernel
from lsst.sims.photUtils.SignalToNoise import *
from lsst.sims.photUtils.PhotometricParameters import PhotometricParameters
from lsst.sims.photUtils.BandpassDict import BandpassDict

def get_epistemic_sigma2(pred_means):
    """
    Parameters
    ----------
    pred_means : np.ndarray of shape [n_MC, n_val, Y_dim]
        Network predictions of the parameter means
    """
    ep_sigma2 = np.var(pred_means, axis=0)
    return ep_sigma2

def get_aleatoric_sigma2(pred_logvar, Y_mean, Y_std):
    """
    Parameters
    ----------
    pred_logvar : np.ndarray of shape [n_MC, n_val, Y_dim]
        Network predictions of the log(parameter sigmas)
    """
    n_Mc, n_val, Y_dim = pred_logvar.shape
    exponentiated = np.exp(pred_logvar)
    al_sigma2 = np.mean(exponentiated, axis=0)
    return al_sigma2

def get_astrometric_error(mag, band, n_visits=184):
    """
    Calculate the astrometric error
    
    Parameters
    ----------
    m5 : float
        5-sigma depth (default: 24.35, r-band for 1 visit)
    n_visits : int
        number of visits (default: 184, mean number of visits in r, i bands)
        
    Returns
    -------
    The astrometric error in as.
    """
    m5 = dict(zip(list('ugrizy'), [23,78, 24.81, 24.35, 23.92, 23.34, 22.45]))
    astrom_err = calcAstrometricError(mag, m5[band], nvisit=n_visits)/1000.0 # mas --> as
    return astrom_err

def get_photometric_error(mag, band, run=None, gamma=None, coadd=True):
    """
    Calculate the photometric error
    
    Returns
    -------
    The magnitude error in mag.
    """
    bandpass_dir = '../throughputs/baseline' # hardcoded for my own system; not necessary for desc-stack
    LSST_BandPass = BandpassDict.loadTotalBandpassesFromFiles(bandpassDir=bandpass_dir)
    bandpass_obj = LSST_BandPass[band]
    phot_params_obj = PhotometricParameters()
    
    # 5-sigma depth
    if coadd:
        m5 = dict(zip(list('ugrizy'), [26.1, 27.4, 27.5, 26.8, 26.1, 24.9]))
    else: 
        # single visit (2 x 15s), for point sources
        # from Ivezic et al 2018
        m5 = dict(zip(list('ugrizy'), [23,78, 24.81, 24.35, 23.92, 23.34, 22.45]))
    # Override if run is provided, meaning it's DC2
    if run == '1.2i':
        m5 = dict(zip(list('ugrizy'), [25.7, 25.7, 25.7, 25.7, 25.7, 25.7])) #FIXME: only r correct for Run 1.2i
        
    mag_err, gamma = calcMagError_m5(mag, bandpass_obj, m5[band], phot_params_obj, gamma)
    return mag_err

def assign_obs_error(param, truth_mag, band, run):
    """
    Assign errors to Object catalog quantities
    
    Returns
    -------
    obs_err : float or np.array
        The error values in units defined in get_astrometric_error(), get_photometric_error
    err_type : str
        Type of observational error
    
    """
    if param in ['ra_offset', 'dec_offset', 'Ixx_sqrt', 'Iyy_sqrt', 'x', 'y_obs',]:
        obs_err = get_astrometric_error(truth_mag, band=band)
        err_type = 'astrometric'
    elif param in ['Ixy', 'IxxPSF', 'IxyPSF', 'IyyPSF',]:
        # \delta(x^2) = \delta(x) \times 2x 
        obs_err = 2.0*param_val*get_astrometric_error(truth_mag, band=band) 
        err_type = 'astrometric'
    elif 'Flux' in param: # flux columns
        obs_err = get_photometric_error(truth_mag, band=band, run=run)
        err_type = 'photometric'
    elif param == 'extendedness':
        obs_err = np.zeros_like(param_val)
        err_type = 'N/A'
    else:
        raise NotImplementedError
    return obs_err, err_type

if __name__=='__main__':
    
    # Quick validation
    
    ref_r_mags = [21, 22, 23, 24]
    ref_coadd_photom = [5, 5, 6, 9] # in mmag, from Ivezic et al 2018
    ref_visit_astrom = [11, 15, 31, 74] # in mas, from Ivezic et al 2018
    ref_coadd_astrom = 10
    
    for i, r in enumerate(ref_r_mags):
        print("r-band mag:", r)
        print("Returned astrometric error (mas) for visit: %.2f" %get_astrometric_error(r, m5=24.35, n_visits=1)*1000.0) # as --> mas
        print("Expected from SRD: %.2f" %ref_visit_astrom[i])
        print("Returned astrometric error (mas) for coadd: %.2f" %get_astrometric_error(r, m5=24.35, n_visits=184)*1000.0) # as --> mas
        print("Expected from SRD: %.2f" %ref_coadd_astrom)
        print("Returned photometric error (mmag) for coadd: %.4f" %get_photometric_error(r, band='r', coadd=True, run='1.2i')*1000.0) # mag --> mmag
        print("Expected from SRD: %.4f" %ref_coadd_photom[i])