import pyMT.data_structures as DS
import pyMT.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


MU = 4 * np.pi * 1e-7


def compute_bost1D(site, method='phase', comp=None, filter_width=1):
    if not comp:
        comp = 'det'
    if 'bost' in comp.lower():
        comp = comp.lower().replace('bost', '')
    rho = utils.compute_rho(site, calc_comp=comp)[0]
    # Flag bad points
    idx = np.array(rho != 0) * (site.used_error['ZXYR'] != site.REMOVE_FLAG)
    rho = rho[idx]
    periods = site.periods[idx]
    # coefs = np.polyfit(np.log10(periods), np.log10(rho), 4)
    # f = np.poly1d(coefs)
    # rhofit = 10 ** f(np.log10(periods))
    if len(rho) != 0:
        rhofit = utils.geotools_filter(periods, rho, fwidth=filter_width)
    else:
        print('No valid data points for bost1D.')
        return [0], [0], [0], [0]
    # log_rho = np.log10(rho)
    log_rho = np.log10(rhofit)
    log_freq = np.log10(periods)
    # log_freq = np.log10(1/periods)
    depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
    # depth = np.sqrt(rho * periods / (2 * np.pi * MU)) / 1000
    if method.lower() == 'bostick':
        slope = np.diff(log_rho) / (np.diff(log_freq))
        slope = list(slope)
        slope.append((log_rho[-1] - log_rho[-2]) / abs(log_freq[-1] - log_freq[-2]))
        slope = np.array(slope)
        # slope[abs(slope) > 0.95] = np.sign(slope[abs(slope) > 0.95]) * 0.95
        bostick = rhofit * ((1 + slope) / (1 - slope))
        # bostick = rho * ((1 + slope) / (1 - slope))
        phase = utils.compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phasefit = utils.geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
    elif method.lower() == 'phase-bostick':
        phase = utils.compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phasefit = utils.geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
        slope = 1 - phasefit / 45
        bostick = rhofit * ((1 + slope) / (1 - slope))
        # slope = 1 - phase / 45
        # bostick = rho * ((1 + slope) / (1 - slope))
    elif method.lower() == 'phase':
        phase = utils.compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phasefit = utils.geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
        bostick = rhofit * ((np.pi / (2 * np.deg2rad(phasefit % 90))) - 1)
        # bostick = rho * ((np.pi / (2 * np.deg2rad(phase % 90))) - 1)
    elif method.lower() == 'niblett':
        cond = 1 / rhofit
        slope = np.diff(cond) / np.diff(depth)
        slope = np.hstack((slope, slope[-1]))
        bostick = 1 / (depth * slope + cond)

        phase = utils.compute_phase(site, calc_comp=comp)[0]
        phase = phase[idx]
        phasefit = utils.geotools_filter(periods, phase, use_log=False, fwidth=filter_width)
    return bostick, depth, rhofit, rho, phasefit, phase

def bost_by_mixed_phimax(site):
    periods = site.periods
    phase = np.array([np.arctan(site.CART[ii].phi_min) for ii in range(site.NP)])
    # phase = np.array([np.arctan(site.CART[ii].phi_max) for ii in range(site.NP)])
    rho   = np.array([site.CART[ii].Ua_phi_max for ii in range(site.NP)])
    rhofit = utils.geotools_filter(site.periods, rho)
    depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
    phasefit = utils.geotools_filter(site.periods, phase, use_log=False, fwidth=1)
    # bostick = rhofit * ((np.pi / (2 * (phasefit % 90))) - 1)
    slope = 1 - phase / 45
    return bostick, depth, rho, phase, rhofit, phasefit

def bost_by_mixed_phimin(site):
    periods = site.periods
    phase = np.array([np.arctan(site.CART[ii].phi_max) for ii in range(site.NP)])
    # phase = np.array([np.arctan(site.CART[ii].phi_max) for ii in range(site.NP)])
    rho   = np.array([site.CART[ii].Ua_phi_min for ii in range(site.NP)])
    rhofit = utils.geotools_filter(site.periods, rho)
    depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
    phasefit = utils.geotools_filter(site.periods, phase, use_log=False, fwidth=1)
    bostick = rhofit * ((np.pi / (2 * (phasefit % 90))) - 1)
    return bostick, depth, rho, phase, rhofit, phasefit

# def bost_by_mixed_phimin(site):
#   periods = site.periods
#   phase = np.array([np.arctan(site.phase_tensors[ii].phi_max) for ii in range(site.NP)])
#   # phase = np.array([np.arctan(site.CART[ii].phi_max) for ii in range(site.NP)])
#   rho   = np.array([site.CART[ii].Ua_phi_min for ii in range(site.NP)])
#   rhofit = utils.geotools_filter(site.periods, rho)
#   depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
#   phasefit = utils.geotools_filter(site.periods, phase, use_log=False, fwidth=1)
#   bostick = rhofit * ((np.pi / (2 * (phasefit % 90))) - 1)
#   return bostick, depth, rho, phase, rhofit, phasefit

def bost_by_cart(site, component='phi_2', method='bostick'):
    phase, phasefit = None, None
    idx = site.used_error['ZXYR'] != site.REMOVE_FLAG
    periods = site.periods[idx]
    rho   = np.array([getattr(site.CART[ii], 'Ua_{}'.format(component)) for ii in range(site.NP)])
    rho   = rho[idx]
    rhofit = utils.geotools_filter(periods, rho)
    depth = np.sqrt(rhofit * periods / (2 * np.pi * MU)) / 1000
    log_rhofit = np.log10(rhofit)
    if method == 'bostick':
        slope = (np.diff(log_rhofit) / np.diff(np.log10(periods)))
        slope = np.hstack((slope, slope[-1]))
        bostick = rhofit * (1 + slope) / (1 - slope)
    elif method == 'niblett':
        cond = 1 / rhofit
        slope = np.diff(cond) / np.diff(depth)
        slope = np.hstack((slope, slope[-1]))
        bostick = 1 / (depth * slope + cond)
    return bostick, depth, rho, rhofit


def bost_by_rotation_max(site):
    def rotz(zxx, zxy, zyx, zyy, theta):
        theta = 2 * (theta)  # Make use of double angle identities
        return ((zxx + zyy) + (zxx - zyy) * np.cos(theta) + (zxy + zyx) * np.sin(theta)) / 2
    zxx = site.data['ZXXR'] + 1j * site.data['ZXXI']
    zxy = site.data['ZXYR'] + 1j * site.data['ZXYI']
    zyx = site.data['ZYXR'] + 1j * site.data['ZYXI']
    zyy = site.data['ZYYR'] + 1j * site.data['ZYYI']
    theta = np.array([pt.azimuth for pt in site.phase_tensors])
    theta = utils.geotools_filter(site.periods, theta, use_log=False)
    zxxR = rotz(zxx, zxy, zyx, zyy, theta)
    zxyR = rotz(zxy, -zxx, zyy, -zyx, theta)
    zyxR = rotz(zyx, zyy, -zxx, -zxy, theta)
    zyyR = rotz(zyy, -zyx, -zxy, zxx, theta)
    new_site = deepcopy(site)
    new_site.data['ZXXR'] = np.real(zxxR)
    new_site.data['ZXXI'] = np.imag(zxxR)
    new_site.data['ZXYR'] = np.real(zxyR)
    new_site.data['ZXYI'] = np.imag(zxyR)
    new_site.data['ZYXR'] = np.real(zyxR)
    new_site.data['ZYXI'] = np.imag(zyxR)
    new_site.data['ZYYR'] = np.real(zyyR)
    new_site.data['ZYYI'] = np.imag(zyyR)
    bostick, depth, rhofit, phase = utils.compute_bost1D(site=new_site, method='phase', comp='xy')
    return bostick, depth, rhofit, phase, rhofit, None

def bost_by_rotation_min(site):
    def rotz(zxx, zxy, zyx, zyy, theta):
        theta = 2 * (theta)  # Make use of double angle identities
        return ((zxx + zyy) + (zxx - zyy) * np.cos(theta) + (zxy + zyx) * np.sin(theta)) / 2
    zxx = site.data['ZXXR'] + 1j * site.data['ZXXI']
    zxy = site.data['ZXYR'] + 1j * site.data['ZXYI']
    zyx = site.data['ZYXR'] + 1j * site.data['ZYXI']
    zyy = site.data['ZYYR'] + 1j * site.data['ZYYI']
    theta = np.array([pt.azimuth for pt in site.phase_tensors])
    theta = utils.geotools_filter(site.periods, theta, use_log=False)
    zxxR = rotz(zxx, zxy, zyx, zyy, theta)
    zxyR = rotz(zxy, -zxx, zyy, -zyx, theta)
    zyxR = rotz(zyx, zyy, -zxx, -zxy, theta)
    zyyR = rotz(zyy, -zyx, -zxy, zxx, theta)
    new_site = deepcopy(site)
    new_site.data['ZXXR'] = np.real(zxxR)
    new_site.data['ZXXI'] = np.imag(zxxR)
    new_site.data['ZXYR'] = np.real(zxyR)
    new_site.data['ZXYI'] = np.imag(zxyR)
    new_site.data['ZYXR'] = np.real(zyxR)
    new_site.data['ZYXI'] = np.imag(zyxR)
    new_site.data['ZYYR'] = np.real(zyyR)
    new_site.data['ZYYI'] = np.imag(zyyR)
    bostick, depth, rhofit, phase = utils.compute_bost1D(site=new_site, method='phase', comp='yx')
    return bostick, depth, rhofit, phase, rhofit, None

# def bost_by_mixed(site):
#     pass

def bost_by_comp(site, comp='xy', method='phase'):
    return compute_bost1D(site=site, method=method, comp=comp)

listfile = 'G:/Other computers/My Computer/sync/Regions/swayze/edi_interpolated/transect_only/transect_only.lst'
datafile = 'G:/Other computers/My Computer/sync/Regions/swayze/gofem/model_transectOnly/swz_reg_Z_transOnly.gdat'
data = DS.Data(listfile=listfile, datafile=datafile)
site = data.sites[data.site_names[0]]

fig = plt.figure()
axes = fig.subplots(nrows=1, ncols=1)

colors = ('r', 'b', 'k')
markers = ('.')#, '-.', '-.')
methods = ('bostick') #, 'phase', 'niblett')
components = ('xy', 'yx', 'ssq')
idx = site.used_error['ZXYR'] != site.REMOVE_FLAG
periods = site.periods[idx]
# for ii, method in enumerate(('xy', 'yx', 'ssq')):
for (color, component) in zip (colors, components):
    bost, depth, rhofit, rho, phasefit, phase = bost_by_comp(site, comp=component, method=methods)
    axes.plot(np.log10(depth), np.log10(bost), marker=markers, color=color, label=component, linestyle='--')
    # axes[1].plot(np.log10(periods), np.log10(rhofit), '--', color=marker, label=method)
    # axes[1].plot(np.log10(periods), np.log10(rho), 'o', color=marker, label=method)
    # axes[2].plot(np.log10(periods), np.log10(phasefit), '--', color=marker, label=method)
    # axes[2].plot(np.log10(periods), np.log10(phase), 'o', color=marker, label=method)



colors = ['r', 'b', 'k']
markers = ['o', 'v']
components = ('phi_max', 'phi_min', 'phi_2')
methods    = ('bostick') #, 'niblett')
           #bost_by_mixed_phimax, bost_by_mixed_phimin,
           # bost_by_cart_max, bost_by_cart_min)
#            #bost_by_rotation_max, bost_by_rotation_min)
# for (marker, method) in zip(markers, methods):
for (color, component) in zip(colors, components):
    # for (marker, method) in zip(markers, methods):
#     bost, depth, rho, phase, rhofit, phasefit = method(site)
    bost, depth, rho, rhofit = bost_by_cart(site, method=methods, component=component)
    axes.plot(np.log10(depth), np.log10(bost), color=color, marker=markers[1], label='CART {}'.format(component))
    # axes[1].plot(np.log10(site.periods), np.log10(rho), 'o', label='rho {}'.format(method.__name__))
#     axes[1].plot(np.log10(site.periods), np.log10(rhofit), '--', label='rhofit {}'.format(method.__name__))
#     # if np.any(phase):
#     #     axes[1,1].plot(np.log10(site.periods), np.rad2deg(phase), 'o', label='phase {}'.format(method.__name__))
#     # if np.any(phasefit):
#     #     axes[1,1].plot(np.log10(site.periods), np.rad2deg(phasefit), 'o', label='phasefit {}'.format(method.__name__))

# for ax in axes.flatten():
    # ax.legend()
axes.legend()
plt.show()
