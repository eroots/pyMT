import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np


listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsites.lst'
dataset = WSDS.Dataset(listfile=listfile)
rmsites = [site for site in dataset.raw_data.site_names if site[0] == 'e' or site[0] == 'd']
dataset.remove_sites(rmsites)
dataset.sort_sites(order='west-east')
rho = {site.name: utils.compute_rho(site)[0] for site in dataset.raw_data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in dataset.raw_data.sites.values()}
bost = {site.name: utils.compute_bost1D(site)[0] for site in dataset.raw_data.sites.values()}
depths = {site.name: utils.compute_bost1D(site)[1] for site in dataset.raw_data.sites.values()}
periods = []
loc = []
rho2 = []
phavals = []
depths2 = []
bost2 = []
for site in dataset.data.site_names:
    for ii, p in enumerate(dataset.raw_data.sites[site].periods):
        periods.append(p)
        bost2.append(bost[site][ii])
        depths2.append(depths[site][ii])
        rho2.append(rho[site][ii])
        phavals.append(pha[site][ii])
        loc.append(dataset.raw_data.sites[site].locations['Y'])
phavals = np.array(phavals)
rho2 = np.array(rho2)
depths2 = np.array(depths2)
bost2 = np.array(bost2)
rhovals = np.log10(rho2)
bostvals = np.log10(bost2)
periods = np.array(periods)
periods = np.log10(periods)
locs = np.array(loc)
points = np.transpose(np.array((locs, periods)))

min_x, max_x = (min(loc), max(loc))
min_p, max_p = (min(periods), max(periods))

grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, 500), np.log10(np.logspace(min_p, max_p, 500)))
grid_rho = griddata(points, phavals, (grid_x, grid_y), method='cubic')
plt.figure()
plt.pcolor(grid_x, grid_y, grid_rho.T)
plt.show()
plt.colorbar()
plt.clim([0, 90])
plt.gca().invert_yaxis()
