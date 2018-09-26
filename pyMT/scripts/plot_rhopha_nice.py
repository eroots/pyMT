import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
import numpy as np


list_file = r'F:\ownCloud\Metal Earth\Data\WinGLinkEDIs\BB.lst'

data = WSDS.RawData(list_file)
sites = ['GER007M', 'ROU006M']
axes = []
fig = plt.figure(figsize=(12, 14))
axes.append(plt.subplot2grid((3, 2), (0, 0), rowspan=2))
axes.append(plt.subplot2grid((3, 2), (2, 0), rowspan=1))
axes.append(plt.subplot2grid((3, 2), (0, 1), rowspan=2))
axes.append(plt.subplot2grid((3, 2), (2, 1), rowspan=1))
# site = data.sites[sites[0]]
for ii, name in enumerate(sites):
    site = data.sites[name]
    rhoxy = utils.compute_rho(site, calc_comp='rhoxy')[0]
    rhoyx = utils.compute_rho(site, calc_comp='rhoyx')[0]
    phaxy = utils.compute_phase(site, calc_comp='phaxy', wrap=True)[0]
    phayx = utils.compute_phase(site, calc_comp='phayx', wrap=True)[0]
    # axes[(ii) * 2].plot(np.log10(site.periods), np.log10(rhoxy), 'bo', markersize=5)
    # axes[(ii) * 2].plot(np.log10(site.periods), np.log10(rhoyx), 'ro', markersize=5)
    axes[(ii) * 2].loglog((site.periods), (rhoxy), 'bo', markersize=5, label='XY')
    axes[(ii) * 2].loglog((site.periods), (rhoyx), 'ro', markersize=5, label='YX')
    axes[(ii) * 2].set_xlabel('Period (s)', fontsize=14)
    axes[(ii) * 2].set_ylabel(r'Apparent Resistivity (${\Omega}$-m)', fontsize=14)
    axes[(ii) * 2].legend(loc=1)
    axes[(ii) * 2].tick_params(axis='both', labelsize=12)
    axes[(ii) * 2].set_title(name, fontsize=16)
    # axes[(ii) * 2].set_ylim([1e0, 1e5])
    axes[(ii) * 2 + 1].semilogx((site.periods), (phaxy), 'bo', markersize=5, label='XY')
    axes[(ii) * 2 + 1].semilogx((site.periods), (phayx), 'ro', markersize=5, label='YX')
    axes[(ii) * 2 + 1].set_xlabel('Period (s)', fontsize=14)
    axes[(ii) * 2 + 1].set_ylabel(r'Degrees (${\degree}$)', fontsize=14)
    axes[(ii) * 2 + 1].legend(loc=1)
    axes[(ii) * 2 + 1].set_ylim([0, 180])
    axes[(ii) * 2 + 1].tick_params(axis='both', labelsize=12)

fig.tight_layout()
plt.savefig('F:/ownCloud/Documents/SoFW2018/Figure_2.png', dpi=300)
plt.show()
