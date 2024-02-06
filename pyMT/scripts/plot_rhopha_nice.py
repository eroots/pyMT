import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np


local_path = 'E:'
# list_file = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/WinGLinkEDIs_final/plot.lst'
# list_file = 'C:/Users/eric/ownCloud/data/Regions/swayze/j2/all.lst'
# list_file = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/swayze/j2/SWZall.lst'
list_file = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/test.lst'
data = WSDS.RawData(list_file)
data_sets = [data]
list_file = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/test.lst'
data = data = WSDS.RawData(list_file)
data_sets.append(data)
marker_set = ['o', 'x']
label_set = ['Orig', 'Interp']
axes = []
sites = data.site_names[:2]
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
for dd, data in enumerate(data_sets):
    marker = marker_set[dd]
    label = label_set[dd]
    for ii, name in enumerate(sites):
        site = data.sites[name]
        rhoxy, rhoxy_err, rhoxy_log10err = utils.compute_rho(site, calc_comp='rhoxy', errtype='errors')
        rhoyx, rhoyx_err, rhoyx_log10err = utils.compute_rho(site, calc_comp='rhoyx', errtype='errors')
        phaxy, phaxy_err = utils.compute_phase(site, calc_comp='phaxy', wrap=True, errtype='errors')[:2]
        phayx, phayx_err = utils.compute_phase(site, calc_comp='phayx', wrap=True, errtype='errors')[:2]
        rhoxy_err *= 0
        rhoyx_err *= 0
        rhoxy_log10err *= 0
        rhoyx_log10err *= 0
        phaxy_err *= 0
        phayx_err *= 0
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
                                yerr=rhoxy_log10err, marker=marker,
                                linestyle='', color='b',
                                markersize=5, label=label+' XY')
        axes[(ii) * 2].errorbar(np.log10(site.periods), np.log10(rhoyx), xerr=None,
                                yerr=rhoyx_log10err, marker=marker,
                                linestyle='', color='r',
                                markersize=5, label=label+' YX')
        axes[(ii) * 2 + 1].errorbar(np.log10(site.periods), phaxy, xerr=None,
                                    yerr=phaxy_err, marker=marker,
                                    linestyle='', color='b',
                                    markersize=5, label=label+' XY')
        axes[(ii) * 2 + 1].errorbar(np.log10(site.periods), phayx, xerr=None,
                                    yerr=phayx_err, marker=marker,
                                    linestyle='', color='r',
                                    markersize=5, label=label+' YX')
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
        axes[(ii) * 2].set_ylim([0, 5])
        axes[(ii) * 2].set_xlim([-3, 4])
        axes[(ii) * 2 + 1].set_xlim([-3, 4])
        # axes[(ii) * 2 + 1].legend(loc=1)
        axes[(ii) * 2 + 1].set_ylim([0, 120])
        axes[(ii) * 2 + 1].tick_params(axis='both', labelsize=12)

fig.tight_layout()
plt.savefig('E:/phd/NextCloud/Documents/GoldenTriangle/RoughFigures/data_interp_compare.png', dpi=300)
# plt.show()
# 