import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import e_colours.colourmaps as cm


cmap = cm.jet_plus(64)
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\2018-517\allsites.lst'
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\all\515-520.lst')
# listfile = r'C:\Users\eric\phd\Kilauea\ConvertedEDIs\all\allsites.lst'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Test_Models\dimensionality\synthLayer.data'
# data = WSDS.RawData(listfile=listfile)
data = WSDS.Data(datafile='C:/Users/eric/phd/Kilauea/stitched/1_day/Z/Kilauea_may_daily.data')
days = (501, 530)
hour_or_day = 1  # sets the interval in labels, so choose appropriately
n_interp = 300
cax_rho = [0, 4]
# data = WSDS.Data(datafile=datafile)
# rmsites = [site for site in data.site_names if site[0] == 'e' or site[0] == 'd']
# data.remove_sites(rmsites)
# data.sort_sites(order='west-east')
rho = {site.name: utils.compute_rho(site)[0] for site in data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in data.sites.values()}
bost = {site.name: utils.compute_bost1D(site)[0] for site in data.sites.values()}
depths = {site.name: utils.compute_bost1D(site)[1] for site in data.sites.values()}
periods = []
loc = []
rhovals = []
phavals = []
depth_vals = []
bostvals = []
for jj, site in enumerate(data.site_names):
    for ii, p in enumerate(data.sites[site].periods):
        periods.append(p)
        bostvals.append(bost[site][ii])
        depth_vals.append(depths[site][ii])
        rhovals.append(rho[site][ii])
        phavals.append(pha[site][ii])
        # loc.append(data.sites[site].locations['Y'])
        loc.append(jj)
phavals = np.array(phavals)
rhovals = np.log10(np.array(rhovals))
depth_vals = np.array(depth_vals)
bostvals = np.log10(np.array(bostvals))
periods = np.log10(np.array(periods))
locs = np.array(loc)
points = np.transpose(np.array((locs, periods)))
points_d = np.transpose(np.array((locs, depth_vals)))
xticks = np.arange(0, loc[-1], hour_or_day)
xtick_labels = [str(x) for x in np.arange(days[0], days[1] + 1)]

min_x, max_x = (min(loc), max(loc))
min_p, max_p = (min(periods), max(periods))
min_d, max_d = (min(depth_vals), max(depth_vals))
grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp),
                             np.log10(np.logspace(min_p, max_p, n_interp)))
grid_pha = griddata(points, phavals, (grid_x, grid_y), method='linear')
grid_rho = griddata(points, rhovals, (grid_x, grid_y), method='linear')

grid_xd, grid_d = np.meshgrid(np.linspace(min_x, max_x, n_interp),
                              np.log10(np.logspace(min_d, max_d, n_interp)))
grid_bost = griddata(points_d, bostvals, (grid_xd, grid_d), method='cubic')


def plot_pha():
    plt.figure()
    plt.pcolor(grid_x, grid_y, grid_pha, cmap=cmap)
    plt.xticks(xticks, xtick_labels)
    plt.clim([0, 90])
    plt.gca().invert_yaxis()
    cb_pha = plt.colorbar()
    cb_pha.set_clim(0, 90)
    cb_pha.ax.tick_params(labelsize=12)
    cb_pha.set_label(r'Phase ($^{\circ}$)',
                     rotation=270,
                     labelpad=20,
                     fontsize=18)
    plt.xlabel('Hour')
    plt.ylabel(r'$\log_{10}$ Period (s)')


def plot_rho():
    plt.figure()
    plt.pcolor(grid_x, grid_y, grid_rho, cmap=cmap)
    plt.xticks(xticks, xtick_labels)
    plt.clim(cax_rho)
    cb_rho = plt.colorbar()
    cb_rho.set_clim(cax_rho[0], cax_rho[1])
    cb_rho.ax.tick_params(labelsize=12)
    cb_rho.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
                     rotation=270,
                     labelpad=20,
                     fontsize=18)
    plt.gca().invert_yaxis()
    plt.xlabel('Hour')
    plt.ylabel(r'$\log_{10}$ Period (s)')


def plot_bost():
    plt.figure()
    plt.pcolor(grid_xd, grid_d, grid_bost, cmap=cmap)
    plt.xticks(xticks, xtick_labels)
    plt.clim(cax_rho)
    cb_rho = plt.colorbar()
    cb_rho.set_clim(cax_rho[0], cax_rho[1])
    cb_rho.ax.tick_params(labelsize=12)
    cb_rho.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
                     rotation=270,
                     labelpad=20,
                     fontsize=18)
    plt.gca().invert_yaxis()
    plt.xlabel('Hour')
    plt.ylabel(r'Depths (m)')


plot_rho()
plot_pha()
plot_bost()
plt.show()
