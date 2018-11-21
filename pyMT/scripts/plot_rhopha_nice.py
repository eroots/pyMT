import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np


list_file = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/WinGLinkEDIs_final/plot.lst'
# list_file = 'C:/Users/eric/ownCloud/data/Regions/swayze/j2/all.lst'

data = WSDS.RawData(list_file)
sites = ['ATT019M', 'SWZ032M']
axes = []
fig = plt.figure(figsize=(12, 6))
axes.append(plt.subplot2grid((3, 2), (0, 0), rowspan=2))
axes.append(plt.subplot2grid((3, 2), (2, 0), rowspan=1))
axes.append(plt.subplot2grid((3, 2), (0, 1), rowspan=2))
axes.append(plt.subplot2grid((3, 2), (2, 1), rowspan=1))
# site = data.sites[sites[0]]
for ii, name in enumerate(sites):
    site = data.sites[name]
    rhoxy, rhoxy_err, rhoxy_log10err = utils.compute_rho(site, calc_comp='rhoxy', errtype='errors')
    rhoyx, rhoyx_err, rhoyx_log10err = utils.compute_rho(site, calc_comp='rhoyx', errtype='errors')
    phaxy, phaxy_err = utils.compute_phase(site, calc_comp='phaxy', wrap=True, errtype='errors')[:2]
    phayx, phayx_err = utils.compute_phase(site, calc_comp='phayx', wrap=True, errtype='errors')[:2]
    axes[(ii) * 2].errorbar(np.log10(site.periods), np.log10(rhoxy), xerr=None,
                            yerr=rhoxy_log10err, marker='o',
                            linestyle='', color='b',
                            markersize=5, label='XY')
    axes[(ii) * 2].errorbar(np.log10(site.periods), np.log10(rhoyx), xerr=None,
                            yerr=rhoyx_log10err, marker='o',
                            linestyle='', color='r',
                            markersize=5, label='YX')
    axes[(ii) * 2 + 1].errorbar(np.log10(site.periods), phaxy, xerr=None,
                                yerr=phaxy_err, marker='o',
                                linestyle='', color='b',
                                markersize=5, label='XY')
    axes[(ii) * 2 + 1].errorbar(np.log10(site.periods), phayx, xerr=None,
                                yerr=phayx_err, marker='o',
                                linestyle='', color='r',
                                markersize=5, label='YX')
    # axes[(ii) * 2].loglog((site.periods), (rhoyx), 'ro', markersize=5, label='YX')
    # axes[(ii) * 2].set_xlabel('Period (s)', fontsize=14)
    if ii == 0:
        axes[(ii) * 2].set_ylabel(r'Apparent Resistivity (${\Omega}$-m)', fontsize=14)
        axes[(ii) * 2 + 1].set_ylabel(r'Degrees (${\degree}$)', fontsize=14)
    axes[(ii) * 2].legend(loc=1)
    axes[(ii) * 2].tick_params(axis='both', labelsize=12)
    axes[(ii) * 2].set_title(name, fontsize=16)
    axes[(ii) * 2].set_ylim([0, 6])
    # axes[(ii) * 2].set_ylim([1e0, 1e5])
    # axes[(ii) * 2 + 1].semilogx((site.periods), (phaxy), 'bo', markersize=5, label='XY')
    # axes[(ii) * 2 + 1].semilogx((site.periods), (phayx), 'ro', markersize=5, label='YX')
    axes[(ii) * 2 + 1].set_xlabel('Period (s)', fontsize=14)
    
    axes[(ii) * 2 + 1].legend(loc=1)
    axes[(ii) * 2 + 1].set_ylim([0, 180])
    axes[(ii) * 2 + 1].tick_params(axis='both', labelsize=12)

fig.tight_layout()
plt.savefig('C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/data_example.png', dpi=300)
plt.show()
