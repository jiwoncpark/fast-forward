import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
import astropy.units as u
import units_utils as units
import uncertainty_utils as uncertain

def draw_cornerplot(pred, labels=None, fig=None, color='black'):
    n_data, n_params = pred.shape
    plot = corner.corner(pred,
                        color=color, 
                        smooth=1.0, 
                        labels=labels,
                        #show_titles=True,
                        fill_contours=True,
                        bins=30,
                        fig=fig,
                        range=[0.68, 0.95, 0.997]*n_params,
                        hist_kwargs=dict(density=True, ))
    return plot

def plot_sample_corner(fig, X, Y, sample, data_meta):
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    safety_mask = ~emulated.isnull().any(1)
    plot_cols = ['psFlux_%s_mag' %bp for bp in 'giy']
    plot_cols += ['cModelFlux_%s_mag' %bp for bp in 'giy']
    plot_cols += ['Ixx', 'Ixy', 'Iyy']
    X = X.loc[safety_mask, :][plot_cols]
    Y = Y.loc[safety_mask, :][plot_cols]
    reg_dim = Y.shape[1] - 1

    sample = sample[safety_mask, :].reshape(-1, reg_dim)
    sample = pd.DataFrame(sample, index=None, columns=meta['Y_cols'][:-1])


    canvas = draw_cornerplot(corner_2d_pred, fig=canvas, color='tab:blue')
    canvas = draw_cornerplot(corner_2d_obs, fig=canvas, color='tab:orange')
    return fig

def plot_confusion_matrix(fig, X, Y, emulated):
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    truth_colname = 'star'
    observed_colname = 'extendedness'
    
    obs = (Y[observed_colname].values).astype(int)
    em = (emulated[observed_colname].values > 0.5).astype(int)
    count, _, _, _ = ax.hist2d(obs, em, norm=matplotlib.colors.LogNorm(), bins=2, cmap=plt.cm.get_cmap("gray")) #marker='.', alpha=0.2, label='_nolegend_')
    ax.set_xlabel('observed')
    ax.set_ylabel('emulated')
    ax.set_title('stars vs. galaxies')
    ax.set_xticks([0.25, 0.75])
    ax.set_xticklabels(['not star (galaxy)', 'star'])
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(['not star (galaxy)', 'star'])

    neg_x = 0.15 
    pos_x = 0.65
    total = np.sum(count)
    ax.text(pos_x, 0.75, "{} ({:.1f}%)".format(int(count[1, 1]), count[1, 1]/total*100.0), fontsize=16, color='white') # true positives
    ax.text(neg_x, 0.75, "{} ({:.1f}%)".format(int(count[0, 1]), count[0, 1]/total*100.0), fontsize=16, color='white') # false positives
    ax.text(pos_x, 0.25, "{} ({:.1f}%)".format(int(count[1, 0]), count[1, 0]/total*100.0), fontsize=16, color='white') # false negatives
    ax.text(neg_x, 0.25, "{} ({:.1f}%)".format(int(count[0, 0]), count[0, 0]/total*100.0), fontsize=16) # true negatives
    #cbar = plt.colorbar()
    #plt.clim(1000, 60000)
    canvas.draw()
    return canvas

def plot_moment(fig, X, Y, emulated, emulated_second, moment_type, display_uncertainty='aleatoric', run='1.2i', plot_offset=True):
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    safety_mask = np.ones((X.shape[0])).astype(bool) # do not mask

    em = emulated.loc[:, moment_type].values
    em_second = emulated_second.loc[:, moment_type].values
    obs = Y.loc[safety_mask, moment_type].values
    n_observed = X.loc[safety_mask, 'n_obs'].values
    al_sig = emulated.loc[safety_mask, '%s_al_sig' %moment_type].values
    ep_sig = emulated.loc[safety_mask, '%s_ep_sig' %moment_type].values
    obs_nomask = Y.loc[:, moment_type].values # for consistent plot scaling
    if display_uncertainty == 'aleatoric':
        display_sig = al_sig
    elif display_uncertainty == 'epistemic':
        display_sig = ep_sig
    
    offset = (em - obs) # as - as
    offset_second = (em_second - obs)
    truth_mag = X.loc[safety_mask, 'truth_total_mag_r'].values
    if moment_type in ['Ix', 'Iy']:
        obs_err = uncertain.get_astrometric_error(truth_mag, 'r', n_observed) # as
        unit = 'as'
    elif moment_type in ['ra_offset', 'dec_offset']:
        obs_err = uncertain.get_astrometric_error(truth_mag, 'r', n_observed)*1000.0 # mas
        unit = 'mas'
    elif moment_type in ['Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF']:
        #obs_err = 2.0 * np.abs(obs) * uncertain.get_astrometric_error(truth_mag, 'r', n_observed) # as^2
        obs_err = np.zeros_like(obs)
        al_sig *= 2.0 * np.abs(obs) 
        ep_sig *= 2.0 * np.abs(obs)
        unit = 'as^2'
    # Sorting necessary for fill_between plots
    sorted_id = np.argsort(obs)
    sorted_obs = obs[sorted_id]
    sorted_err = obs_err[sorted_id]
    
    # Perfect mapping
    perfect = np.linspace(np.min(obs_nomask), np.max(obs_nomask), 20)
    # Baseline errors
    ax.fill_between(sorted_obs, -sorted_err, sorted_err, alpha=0.5, facecolor='tab:orange', label=r'1-$\sigma$ photometric')
    # Plot estimated uncertainty
    ax.errorbar(obs, offset, color='tab:blue', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'$N_1$ 1-$\sigma$ %s' %display_uncertainty)
    ax.errorbar(obs, offset_second, color='tab:olive', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'N_2 1-$\sigma$ %s' %display_uncertainty)
    # Plot perfect mapping
    ax.plot(perfect, np.zeros_like(perfect), linestyle='--', color='r', label="Perfect mapping")
    #ax.set_ylim([-5, 5])
    ax.set_title(moment_type)
    ax.set_ylabel('Emulated - Observed (%s)' %unit)
    ax.set_xlabel('Observed (%s)' %unit)
    #ax.set_xscale('symlog')
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ epistemic: %.2f (%s)" %(np.mean(ep_sig), unit))
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ aleatoric: %.2f (%s)" %(np.mean(al_sig), unit))
    ax.legend(loc=(1.05, 0.5))
    canvas.draw()
    return canvas

def plot_flux(fig, X, Y, emulated, emulated_second, flux_formatting, bandpass, display_uncertainty='aleatoric', run='1.2i', plot_offset=True):
    
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    flux_name = flux_formatting %bandpass
    safety_mask = np.ones((X.shape[0])).astype(bool) # do not mask
    zoom_factor = 1.e6

    em = emulated.loc[safety_mask, flux_name].values
    em_second = emulated_second.loc[safety_mask, flux_name].values
    obs = Y.loc[safety_mask, flux_name].values
    obs_mag = Y.loc[safety_mask, '%s_mag' %flux_name].values
    al_sig = emulated.loc[safety_mask, '%s_al_sig_flux' %flux_name].values
    ep_sig = emulated.loc[safety_mask, '%s_ep_sig_flux' %flux_name].values
    if display_uncertainty == 'aleatoric':
        display_sig = al_sig * zoom_factor
    elif display_uncertainty == 'epistemic':
        display_sig = ep_sig * zoom_factor
    
    obs_nomask = Y.loc[:, flux_name].values * zoom_factor # for consistent plot scaling
    offset = (em - obs) * zoom_factor # Jy - Jy
    offset_second = (em_second - obs) * zoom_factor
    obs_display = obs * zoom_factor
    #truth_mag = X.loc[safety_mask, 'truth_total_mag_%s' %bandpass].values
    obs_err = uncertain.get_photometric_error(obs_mag, 'r', run=run, gamma=None, coadd=True) # mag
    obs_err = units.delta_flux(flux=obs, delta_mag=obs_err) 
    obs_err_display = obs_err * zoom_factor
    # Sorting necessary for fill_between plots
    sorted_id = np.argsort(obs_display)
    sorted_obs = obs_display[sorted_id]
    sorted_err = obs_err_display[sorted_id]
    
    # Perfect mapping
    perfect = np.linspace(np.min(obs_nomask), np.max(obs_nomask), 20)
    # Baseline errors
    ax.fill_between(sorted_obs, -sorted_err, sorted_err, alpha=0.5, facecolor='tab:orange', label=r'1-$\sigma$ photometric')
    # Plot estimated uncertainty
    ax.errorbar(obs_display, offset, color='tab:blue', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'$N_1$ 1-$\sigma$ %s' %display_uncertainty)
    ax.errorbar(obs_display, offset_second, color='tab:olive', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'$N_2$ 1-$\sigma$ %s' %display_uncertainty)
    # Plot perfect mapping
    ax.plot(perfect, np.zeros_like(perfect), linestyle='--', color='r', label="Perfect mapping")
    #ax.set_ylim([-5, 5])
    ax.set_title(flux_name)
    ax.set_ylabel('Emulated - Observed (microJy)')
    ax.set_xlabel('Observed (microJy)')
    ax.set_xlim( [np.min(obs_nomask), np.max(obs_nomask)] )
    #plt.yscale('symlog')
    ax.set_xscale('symlog')
    #print(param_star)
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ epistemic: %.2f (microJy)" %np.mean(ep_sig))
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ aleatoric: %.2f (microJy)" %np.mean(al_sig))
    ax.legend(loc=(1.05, 0.5))
    canvas.draw()
    return canvas

def plot_magnitude(fig, X, Y, emulated, emulated_second, flux_formatting, bandpass, display_uncertainty='aleatoric', run='1.2i', plot_offset=True):
    
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    flux_name = flux_formatting %bandpass
    safety_mask = ~emulated.loc[:, '%s_mag' %flux_name].isnull()

    em_mag = emulated.loc[safety_mask, '%s_mag' %flux_name].values
    em_mag_second = emulated_second.loc[safety_mask, '%s_mag' %flux_name].values
    obs_mag = Y.loc[safety_mask, '%s_mag' %flux_name].values
    al_sig_mag = emulated.loc[safety_mask, '%s_al_sig_mag' %flux_name].values
    ep_sig_mag = emulated.loc[safety_mask, '%s_ep_sig_mag' %flux_name].values
    obs_nomask = Y.loc[:, '%s_mag' %flux_name].values # for consistent plot scaling
    obs_nomask = obs_nomask[np.isfinite(obs_nomask)]
    if display_uncertainty == 'aleatoric':
        display_sig = al_sig_mag
    elif display_uncertainty == 'epistemic':
        display_sig = ep_sig_mag
    
    offset = (em_mag - obs_mag) # mag - mag
    offset_second = (em_mag_second - obs_mag)
    #truth_mag = X.loc[safety_mask, 'truth_total_mag_%s' %bandpass].values
    obs_err = uncertain.get_photometric_error(obs_mag, 'r', run=run, gamma=None, coadd=True) # mag
    # Sorting necessary for fill_between plots
    sorted_id = np.argsort(obs_mag)
    sorted_obs = obs_mag[sorted_id]
    sorted_err = obs_err[sorted_id]

    # Perfect mapping
    perfect = np.linspace(np.min(obs_nomask), np.max(obs_nomask), 20)
    # Baseline errors
    ax.fill_between(sorted_obs, -sorted_err, sorted_err, alpha=0.5, facecolor='tab:orange', label=r'1-$\sigma$ photometric')
    # Plot estimated uncertainty
    ax.errorbar(obs_mag, offset, color='tab:blue', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'$N_1$ 1-$\sigma$ %s' %display_uncertainty)
    ax.errorbar(obs_mag, offset_second, color='tab:olive', marker='.', linewidth=0, yerr=display_sig, elinewidth=0.5, label=r'$N_2$ 1-$\sigma$ %s' %display_uncertainty)
    # Plot perfect mapping
    ax.plot(perfect, np.zeros_like(perfect), linestyle='--', color='r', label="Perfect mapping")
    #plt.plot(perfect, perfect, linestyle='--', color='r', label="Perfect mapping")
    ax.set_ylim([-5, 5])
    ax.set_title(flux_name)
    ax.set_ylabel('Emulated - Observed (mag)')
    ax.set_xlabel('Observed (mag)')
    ax.set_xlim( [np.min(obs_nomask), min(28.0, np.max(obs_nomask))] )
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ epistemic: %.2f (mag)" %np.mean(ep_sig_mag))
    ax.plot([], [], ' ', label=r"Avg 1-$\sigma$ aleatoric: %.2f (mag)" %np.mean(al_sig_mag))
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend(loc=(1.05, 0.5))
    canvas.draw()
    return canvas

def get_natural_units(X, Y, mu, al_sig2, ep_sig2, F, mu_second, al_sig2_second, ep_sig2_second, F2, alpha, mu_class, al_sig2_class, ep_sig2_class, meta):
    revert_flux = 1.0/meta['scale_flux']
    null_mag_flag = -1
    ref_centroid = meta['ref_centroid']
    
    # For broadcasting
    X_mean = np.array(meta['X_mean']).reshape(1, -1)
    X_std = np.array(meta['X_std']).reshape(1, -1)
    
    # Unstandardize
    X = X*X_std + X_mean

    # Concat classification results
    mu = np.concatenate([mu_class, mu,], axis=1)
    mu_second = np.concatenate([mu_class, mu_second,], axis=1)
    al_sig2 = np.concatenate([al_sig2_class, al_sig2,], axis=1)
    ep_sig2 = np.concatenate([ep_sig2_class, ep_sig2,], axis=1)
    al_sig2_second = np.concatenate([al_sig2_class, al_sig2_second,], axis=1)
    ep_sig2_second = np.concatenate([ep_sig2_class, ep_sig2_second,], axis=1)
    
    # Dictify
    X = pd.DataFrame(X, index=None, columns=meta['X_cols'])
    Y = pd.DataFrame(Y, index=None, columns=meta['Y_cols'])
    mu = pd.DataFrame(mu, index=None, columns=meta['Y_cols'])
    mu_second = pd.DataFrame(mu_second, index=None, columns=meta['Y_cols'])
    al_sig2 = pd.DataFrame(al_sig2, index=None, columns=meta['Y_cols'])
    ep_sig2 = pd.DataFrame(ep_sig2, index=None, columns=meta['Y_cols'])
    al_sig2_second = pd.DataFrame(al_sig2_second, index=None, columns=meta['Y_cols'])
    ep_sig2_second = pd.DataFrame(ep_sig2_second, index=None, columns=meta['Y_cols'])
    
    X.loc[:, 'star'] = X['star'].values.astype(bool)

    # Positions
    Y['ra_obs'] = Y.loc[:, 'ra_offset']/1000.0/3600.0 + X['ra_truth'].values
    Y['dec_obs'] = Y.loc[:, 'dec_offset']/1000.0/3600.0 + X['dec_truth'].values
    mu['ra_obs'] = mu.loc[:, 'ra_offset']/1000.0/3600.0 + X['ra_truth'].values
    mu['dec_obs'] = mu.loc[:, 'dec_offset']/1000.0/3600.0 + X['dec_truth'].values
    
    # Add mag cols
    for bp in 'ugrizy':
        # rescale fluxes
        truth_total_flux = X['%s_flux' %bp].values * revert_flux # Jy
        truth_gal_flux = X.loc[X['star']==False, 'mag_true_%s_lsst' %bp].values * revert_flux # Jy
        X.loc[:, '%s_flux' %bp] = truth_total_flux
        X.loc[X['star']==False, 'mag_true_%s_lsst' %bp] = truth_gal_flux
        # add truth total, gal mags
        X['truth_total_mag_%s' %bp] = (truth_total_flux * u.Jy).to_value(u.ABmag)
        X['truth_gal_mag_%s' %bp] = null_mag_flag # for stars
        X.loc[X['star']==False, 'truth_gal_mag_%s' %bp] = (truth_gal_flux * u.Jy).to_value(u.ABmag)
    
        for flux_type in ['cModelFlux', 'psFlux']:
            # rescale, rezero
            Y['%s_%s_offset' %(flux_type, bp)] = Y['cModelFlux_%s' %bp].values # rename
            mu['%s_%s_offset' %(flux_type, bp)] = mu['%s_%s' %(flux_type, bp)].values
            mu_second['%s_%s_offset' %(flux_type, bp)] = mu_second['%s_%s' %(flux_type, bp)].values
            Y.loc[:, '%s_%s' %(flux_type, bp)] *= revert_flux # Jy
            Y.loc[:, '%s_%s' %(flux_type, bp)] += truth_total_flux
            mu.loc[:, '%s_%s' %(flux_type, bp)] *= revert_flux # Jy
            mu.loc[:, '%s_%s' %(flux_type, bp)] += truth_total_flux
            mu_second.loc[:, '%s_%s' %(flux_type, bp)] *= revert_flux # Jy
            mu_second.loc[:, '%s_%s' %(flux_type, bp)] += truth_total_flux
            # add mags
            Y['%s_%s_mag' %(flux_type, bp)] = (Y['%s_%s' %(flux_type, bp)].values * u.Jy).to_value(u.ABmag)
            mu['%s_%s_mag' %(flux_type, bp)] = (mu['%s_%s' %(flux_type, bp)].values * u.Jy).to_value(u.ABmag)
            mu_second['%s_%s_mag' %(flux_type, bp)] = (mu_second['%s_%s' %(flux_type, bp)].values * u.Jy).to_value(u.ABmag)
    
    for bp in 'ugrizy':
        for flux_type in ['cModelFlux', 'psFlux']:
            # Rescale flux uncertainties, add as column
            al_sig_flux = al_sig2['%s_%s' %(flux_type, bp)].values**0.5 * revert_flux # Jy
            mu['%s_%s_al_sig_flux' %(flux_type, bp)] = al_sig_flux
            ep_sig_flux = ep_sig2['%s_%s' %(flux_type, bp)].values**0.5 * revert_flux # Jy
            mu['%s_%s_ep_sig_flux' %(flux_type, bp)] = ep_sig_flux

            al_sig_flux_second = al_sig2_second['%s_%s' %(flux_type, bp)].values**0.5 * revert_flux # Jy
            mu_second['%s_%s_al_sig_flux' %(flux_type, bp)] = al_sig_flux_second
            ep_sig_flux_second = ep_sig2_second['%s_%s' %(flux_type, bp)].values**0.5 * revert_flux # Jy
            mu_second['%s_%s_ep_sig_flux' %(flux_type, bp)] = ep_sig_flux_second

            # Convert into mag
            em_flux = mu['%s_%s' %(flux_type, bp)].values
            em_mag = mu['%s_%s_mag' %(flux_type, bp)].values
            al_sig_mag = units.delta_mag(flux=em_flux, mag=em_mag, delta_flux=al_sig_flux)
            mu['%s_%s_al_sig_mag' %(flux_type, bp)] = al_sig_mag
            ep_sig_mag = units.delta_mag(flux=em_flux, mag=em_mag, delta_flux=ep_sig_flux)
            mu['%s_%s_ep_sig_mag' %(flux_type, bp)] = ep_sig_mag
            mu['%s_%s_total_sig_flux'] = ep_sig_flux + al_sig_flux
            mu['%s_%s_total_sig_mag'] = (ep_sig_mag**2.0 + al_sig_mag**2.0)**0.5

            em_flux_second = mu_second['%s_%s' %(flux_type, bp)].values
            em_mag_second = mu_second['%s_%s_mag' %(flux_type, bp)].values
            al_sig_mag_second = units.delta_mag(flux=em_flux_second, mag=em_mag_second, delta_flux=al_sig_flux_second)
            mu_second['%s_%s_al_sig_mag' %(flux_type, bp)] = al_sig_mag_second
            ep_sig_mag_second = units.delta_mag(flux=em_flux, mag=em_mag, delta_flux=ep_sig_flux)
            mu_second['%s_%s_ep_sig_mag' %(flux_type, bp)] = ep_sig_mag
            mu_second['%s_%s_total_sig_flux'] = ep_sig_flux_second + al_sig_flux_second
            mu_second['%s_%s_total_sig_mag'] = (ep_sig_mag_second**2.0 + al_sig_mag_second**2.0)**0.5
    
    # Second moments in as^2
    Y.loc[:, ['Ixx', 'Iyy']] *= Y.loc[:, ['Ixx', 'Iyy']].values
    Y.loc[:, ['IxxPSF', 'IyyPSF']] *= Y.loc[:, ['IxxPSF', 'IyyPSF']].values
    mu.loc[:, ['Ixx', 'Iyy']] *= mu.loc[:, ['Ixx', 'Iyy']].values
    mu.loc[:, ['IxxPSF', 'IyyPSF']] *= mu.loc[:, ['IxxPSF', 'IyyPSF']].values
    mu_second.loc[:, ['Ixx', 'Iyy']] *= mu_second.loc[:, ['Ixx', 'Iyy']].values
    mu_second.loc[:, ['IxxPSF', 'IyyPSF']] *= mu_second.loc[:, ['IxxPSF', 'IyyPSF']].values

    for moment_type in ['Ixx', 'Ixy', 'Iyy', 'IxxPSF', 'IxyPSF', 'IyyPSF']:
        mu['%s_al_sig' %moment_type] = np.abs(0.5*mu.loc[:, moment_type].values**(-0.5)) * al_sig2[moment_type].values**0.5
        mu['%s_ep_sig' %moment_type] = np.abs(0.5*mu.loc[:, moment_type].values**(-0.5)) * ep_sig2[moment_type].values**0.5
        mu_second['%s_al_sig' %moment_type] = np.abs(0.5*mu_second.loc[:, moment_type].values**(-0.5)) * al_sig2_second[moment_type].values**0.5
        mu_second['%s_ep_sig' %moment_type] = np.abs(0.5*mu_second.loc[:, moment_type].values**(-0.5)) * ep_sig2_second[moment_type].values**0.5

    # Offset in ra, dec in mas
    for pos in ['ra_offset', 'dec_offset']:
        mu['%s_al_sig' %pos] = al_sig2[pos].values**0.5
        mu['%s_ep_sig' %pos] = ep_sig2[pos].values**0.5
        mu_second['%s_al_sig' %pos] = al_sig2_second[pos].values**0.5
        mu_second['%s_ep_sig' %pos] = ep_sig2_second[pos].values**0.5
                                                        
    return X, Y, mu, mu_second