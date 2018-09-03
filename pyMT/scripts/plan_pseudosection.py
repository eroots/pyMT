import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import naturalneighbor as nn
import matplotlib.pyplot as plt
import numpy as np
import e_colours.colourmaps as cm


cmap = cm.jet_plus(64)
# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsites.lst'
# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\j2\allbb.lst')
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\j2\allbb.data')
datafile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/swz_cull1/swz_cull1f_Z.dat'
data = WSDS.Data(datafile=datafile)
# rmsites = [site for site in data.site_names if site[0] == 'e' or site[0] == 'd']
# data.remove_sites(rmsites)
# data.sort_sites(order='west-east')
rho = {site.name: utils.compute_rho(site)[0] for site in data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in data.sites.values()}
rho_lim = [0, 5]
n_interp = 200
period = 0
# bost = {site.name: utils.compute_bost1D(site)[0] for site in data.sites.values()}
# depths = {site.name: utils.compute_bost1D(site)[1] for site in data.sites.values()}
periods = []
loc_x = []
loc_y = []
rho_vals = []
phase_vals = []
loc_z = []
for dim in (0, 1):
    for site in data.site_names:
        # for ii, p in enumerate(data.sites[site].periods):
            # periods.append(p)
            # bost2.append(bost[site][ii])
            # depths2.append(depths[site][ii])
            rho_vals.append(rho[site][period])
            phase_vals.append(pha[site][period])
            # # Flip the coords here so X is west-east
            loc_y.append(data.sites[site].locations['X'])
            loc_x.append(data.sites[site].locations['Y'])
            loc_z.append(dim)
phase_vals = np.array(phase_vals)
rho_vals = np.array(rho_vals)
# rho_vals = np.log10(rho_vals)
loc_x = np.array(loc_x)
loc_y = np.array(loc_y)
loc_z = np.array(loc_z)
points = np.transpose(np.array((loc_x, loc_y, loc_z)))
# points = np.transpose(np.array((loc_x, loc_y)))

min_x, max_x = (min(loc_x), max(loc_x))
min_y, max_y = (min(loc_y), max(loc_y))

grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp),
                             np.linspace(min_y, max_y, n_interp))
step_size_x = (max_x - min_x) / n_interp
step_size_y = (max_y - min_y) / n_interp
grid_ranges = [[min_x, max_x, step_size_x], [min_y, max_y, step_size_y], [0, 1, 1]]
# grid_rho = griddata(points, rho_vals, (grid_x, grid_y), method='linear')
# grid_pha = griddata(points, phase_vals, (grid_x, grid_y), method='linear')
grid_rho = np.squeeze(nn.griddata(points, rho_vals, grid_ranges))
grid_pha = np.squeeze(nn.griddata(points, phase_vals, grid_ranges))
plt.figure()
plt.pcolor(grid_x, grid_y, np.log10(grid_rho.T), cmap=cmap)
plt.colorbar()
plt.clim(rho_lim)
# plt.gca().invert_yaxis()
plt.plot(loc_x, loc_y, 'k.')
plt.figure()
plt.pcolor(grid_x, grid_y, grid_pha.T, cmap=cmap)
plt.colorbar()
plt.clim([0, 90])
# plt.gca().invert_yaxis()
plt.plot(loc_x, loc_y, 'k.')
plt.show()
