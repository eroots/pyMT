import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np


local_path = 'E:'
# list_file = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/WinGLinkEDIs_final/plot.lst'
# list_file = 'C:/Users/eric/ownCloud/data/Regions/swayze/j2/all.lst'
list_file = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/swayze/j2/SWZall.lst'
data = WSDS.RawData(list_file)
# sites = ['ATT019M', 'SWZ032M']
sites = ['p92001', 'p92002']
# sites = ['SWZ016M', 'SWZ015A', 'SWZ014M', 'SWZ013A']
sites = ['SWZ117M', 'SWZ116A'] #, 'SWZ118A', 'SWZ116A']
axes = []
fig = plt.figure(figsize=(12, 6))
# axes.append(plt.subplot2grid((4, 2), (0, 0), rowspan=2))
# axes.append(plt.subplot2grid((4, 2), (2, 0), rowspan=2))
# axes.append(plt.subplot2grid((4, 2), (0, 1), rowspan=2))
# axes.append(plt.subplot2grid((4, 2), (2, 1), rowspan=2))
# For plotting phases
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
    #########################################################################
    # For multiple sites of Rho only
    # axes[(ii)].errorbar(np.log10(site.periods), np.log10(rhoxy), xerr=None,
    #                         yerr=rhoxy_log10err, marker='o',
    #                         linestyle='', color='b',
    #                         markersize=5, label='XY')
    # axes[(ii)].errorbar(np.log10(site.periods), np.log10(rhoyx), xerr=None,
    #                         yerr=rhoyx_log10err, marker='o',
    #                         linestyle='', color='r',
    #                         markersize=5, label='YX')
    #########################################################################
    # For Rho and smaller phase plots beneath
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
    axes[(ii) * 2 + 1].set_xlabel('Period (s)', fontsize=14)
    if ii == 0:
        axes[(ii) * 2].set_ylabel(r'Apparent Resistivity (${\Omega}$-m)', fontsize=14)
        # axes[(ii) * 2 + 1].set_ylabel(r'Apparent Resistivity (${\Omega}$-m)', fontsize=14)
        axes[(ii) * 2 + 1].set_ylabel(r'Degrees (${\degree}$)', fontsize=14)
        axes[(ii)].legend(loc=1)
    # axes[(ii)].tick_params(axis='both', labelsize=12)
    axes[(ii) * 2].set_title(name, fontsize=16)
    # axes[(ii)].set_ylim([2, 7])
    # axes[(ii)].set_xlim([-2, 1])
    # axes[(ii) * 2 + 1].semilogx((site.periods), (phaxy), 'bo', markersize=5, label='XY')
    # axes[(ii) * 2 + 1].semilogx((site.periods), (phayx), 'ro', markersize=5, label='YX')
# axes[1].set_xlabel('Period (s)', fontsize=14)
# axes[3].set_xlabel('Period (s)', fontsize=14)
    axes[(ii) * 2].set_ylim([2, 6.5])
    axes[(ii) * 2].set_xlim([-2, 0])
    axes[(ii) * 2 + 1].set_xlim([-2, 0])
    # axes[(ii) * 2 + 1].legend(loc=1)
    axes[(ii) * 2 + 1].set_ylim([0, 90])
    axes[(ii) * 2 + 1].tick_params(axis='both', labelsize=12)

fig.tight_layout()
# plt.savefig('C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/data_example.png', dpi=300)
plt.show()
