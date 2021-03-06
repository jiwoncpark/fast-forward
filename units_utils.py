import numpy as np
import astropy.units as u

def delta_flux(flux, delta_mag):
    return np.log(10.0)/2.5*flux*delta_mag

def delta_mag(mag, flux, delta_flux):
    """
    mag: ABmag
    flux: Jy
    delta_flux: Jy
    """
    return 2.5/np.log(10.0)*delta_flux/flux

def nmgy_to_njy(from_flux, inv=False):
    nmgy_to_njy = 3613.0 # 1 nMgy = 3.613e-6 Jy = 3613 nJy
    if inv: # njy --> nmgy
        return from_flux*nmgy_to_njy
    else: # nmgy --> njy
        return from_flux/nmgy_to_njy

def flux_to_mag(flux, zeropoint_mag=0.0, from_unit=None, to_unit=None):
    if from_unit=='nMgy':
        zeropoint_mag=22.5
    return zeropoint_mag-2.5*np.log10(flux)

def mag_to_flux(mag, zeropoint_mag=0.0, from_unit=None, to_unit=None):
    if to_unit=='nMgy':
        zeropoint_mag=22.5
    return np.power(10.0, -0.4*(mag - zeropoint_mag))

def fwhm_to_sigma(fwhm):
    return fwhm/np.sqrt(8.0*np.log(2.0))

def deg_to_arcsec(deg):
    return 3600.0*deg

def arcsec_to_deg(arcsec):
    return arcsec/3600.0

def e1e2_to_phi(e1, e2):
    phi = 0.5*np.arctan(e2/e1)
    return phi

def e1e2_to_ephi(e1, e2):
    e = np.power(np.power(e1, 2.0) + np.power(e2, 2.0), 0.5)
    phi = 0.5*np.arctan(e2/e1)
    return e, phi