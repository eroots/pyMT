import numpy as np
import cma
from pyMT.utils import compute_ssq, compute_MT_determinant


mu = 4 * np.pi * 1e-7

def compute_rho(Z, periods):
    return np.real(Z * np.conj(Z) * periods / (mu * 2 * np.pi))

def compute_phase(Z):
    return np.rad2deg(np.arctan2(np.imag(Z), np.real(Z)))

def compute_det1D(Z):
    # Determinant and SSQ determinant are equivalent in 1D
    return np.sqrt((Z * np.conj(Z)))

def wait_1D(periods, model, dz):
    """
    Takes a model and periods and returns the 1-D impedance response
    Inputs:
        periods: A NPx1 numpy array containing the periods to generate the resposne at
        model: A Mx1 numpy array containing the resistivity values of the 1D model
        dz: A M+1x1 numpy array containing the interface depths of the model
    Outputs:
        z_obs: A NPx1 array containing the 1D impedance response for the given model at the given periods
    """
    cond = 1 / np.array(model)
    omega = 2 * np.pi / periods
    scale = 1 / (4 * np.pi / 10000000)
    z_obs = np.zeros(len(periods), dtype=np.complex128)
    rhoa = np.zeros(len(periods))
    phi = np.zeros(len(periods))
    for nfreq, w in enumerate(omega):
        prop_const = np.sqrt(1j*mu*cond[-1] * w)
        C = np.zeros(len(cond), dtype=np.complex128)
        C[-1] = 1 / prop_const
        if len(dz) > 1:
            for k in reversed(range(len(cond) - 1)):
                prop_layer = np.sqrt(1j*w*mu*cond[k])
                k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * dz[k]))
                k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * dz[k])) + 1)
                C[k] = np.squeeze((1 / prop_layer) * (k1 / k2))
        z_obs[nfreq] = 1j * w * mu * C[0]

    return z_obs


def objective_function(model, *args):
    """
    Calculates the objective function for a given model and set of arguments
    """
    (data_obs, err_obs, periods, dz, data_norm, model_norm, regpar, target_misfit, mantle_transition) = args
    # rhoa_obs = compute_rho(compute_det1D(data_obs), periods)
    # model = 10**model
    # if mantle_transition:
    #     idx = np.argmin(410000 - np.array(dz))
    #     model[idx:] = np.log10(20)
    rhoa_mod = compute_rho(compute_det1D(wait_1D(periods, 10**model, dz)), periods)
    rhoa_mod = np.log10(rhoa_mod)
    # data_obs = np.log10(data_obs)
    # rhoa_mod = compute_rho(compute_det1D(z_mod), periods)
    # So far only rhoa modelling is done
    # pha_mod = compute_phase(compute_det1D(z_mod))
    reg_term = 0.
    for i in range(len(model)-1):
        reg_term += abs(model[i+1]-model[i])**model_norm

    misfit = np.sqrt(np.mean(((rhoa_mod - data_obs)**2) / err_obs**2))
    phi = (1/regpar) * np.abs(misfit - target_misfit) + reg_term

    return phi

def _invert_1DMT(rhoa_obs, err_obs, periods, dz, rho_initial,
                regpar=20, data_norm=2, model_norm=2, rho_min=1e-2,
                rho_max=1e6, maxiter=100, target_misfit=1.0, mantle_transition=False):

    x0 = np.log10(rho_initial)
    sigma0 = 0.2
    args = (rhoa_obs, err_obs, periods, dz, data_norm, model_norm, regpar, target_misfit, mantle_transition)
    options = {'bounds':[np.log10(rho_min), np.log10(rho_max)], 'maxiter':maxiter,
               'tolfunhist': 1e-2}
    if mantle_transition:
        idx = np.argmin(409999 - np.array(dz))
        x0[idx-1:] = np.log10(20)
        options.update({'fixed_variables': {ii: np.log10(20) for ii in range(idx-1, len(x0))}})
    final_model, es = cma.fmin2(objective_function, x0, sigma0, options, args=args)
    final_model = 10**final_model
    z_best = wait_1D(periods, final_model, dz)
    rhoa_best = np.log10(compute_rho(compute_det1D(z_best), periods))
    rms_best = np.sqrt((np.abs(rhoa_best - rhoa_obs)**2 / err_obs**2).mean())

    return final_model, rhoa_best, z_best, rms_best

def invert_1DMT(data, rho_initial, dz,
                mode='ssq', uncertainty=0.1, regpar=0.1, data_norm=2,
                model_norm=2, rho_min=1e-2, rho_max=1e6,
                maxiter=100, target_misfit=1.0, mantle_transition=False):
    """
        Wraps _invert_1DMT to take a pyMT Data structure as its input
    """
    _, rhoa_obs = data.average_rho(comp=mode)
    rhoa_obs = np.log10(rhoa_obs)
    # rho_initial = np.array(list(rho_initial) + [hs])
    err_obs = np.ones(shape=rhoa_obs.shape) * uncertainty

    final_model, rhoa_best, z_best, rms_best = _invert_1DMT(rhoa_obs, err_obs, data.periods, dz, rho_initial,
                                                            regpar, data_norm, model_norm, rho_min,
                                                            rho_max, maxiter, target_misfit, mantle_transition)
    rhoa_best = 10**rhoa_best
    return final_model, rhoa_best, z_best, rms_best


