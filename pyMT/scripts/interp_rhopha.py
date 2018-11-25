import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import e_colours.colourmaps as cm
import naturalneighbor as nn


# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsites.lst'
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Test_Models\dimensionality\synthLayer.data'
listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst'
# data = WSDS.Data(datafile=datafile)
data = WSDS.RawData(listfile)
cmap = cm.jet()
n_interp = 300
# rmsites = [site for site in data.site_names if site[0] == 'e' or site[0] == 'd']
# data.remove_sites(rmsites)
# data.sort_sites(order='west-east')
data.locations = data.locations[data.locations[:, 0].argsort()]  # Make sure they go north-south
# A little kludge to make sure the last few sites are in the right order (west-east)
data.locations[0:8, :] = data.locations[np.flip(data.locations[0:8, 1].argsort())]
rho = {site.name: utils.compute_rho(site)[0] for site in data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in data.sites.values()}
bost = {site.name: utils.compute_bost1D(site)[0] for site in data.sites.values()}
depths = {site.name: utils.compute_bost1D(site)[1] for site in data.sites.values()}
periods = []
loc = []
rho2 = []
phavals = []
depths2 = []
bost2 = []
for site in data.site_names:
    for ii, p in enumerate(data.sites[site].periods):
        if p in data.narrow_periods.keys():
            if data.narrow_periods[p] > 0.9:
                periods.append(p)
                bost2.append(bost[site][ii])
                depths2.append(depths[site][ii])
                rho2.append(rho[site][ii])
                phavals.append(pha[site][ii])
                loc.append(data.sites[site].locations['X'])
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

grid_ranges = [[min_x, max_x, n_interp * 1j],
               [min_p, max_p, n_interp * 1j],
               [0, 1, 1]]
grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp), np.log10(np.logspace(min_p, max_p, 500)))
# grid_vals = griddata(points, phavals, (grid_x, grid_y), method='linear')
grid_vals = np.squeeze(nn.griddata(points, phavals, grid_ranges))
plt.figure()
plt.pcolor(grid_x, grid_y, grid_vals, cmap=cmap)
plt.colorbar()
plt.clim([0, 90])
plt.gca().invert_yaxis()
plt.show()
