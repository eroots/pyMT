import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import naturalneighbor as nn
import matplotlib.pyplot as plt
import numpy as np
import e_colours.colourmaps as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


# cmap = cm.jet_plus(64)
cmap = cm.bwr(64)
# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsites.lst'
# listfile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\j2\allbb.lst')
# datafile = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\j2\allbb.data')
# datafile = 'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/swz_cull1/swz_cull1f_Z.dat'
# datafile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/cull_allSuperior.data'
# listfile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/culled_allSuperior.lst'
datafile = 'C:/Users/eroots/phd/ownCloud/data/Regions/afton/sorted_lines.dat'
listfile = 'C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/sorted_lines.lst'
data = WSDS.Data(datafile=datafile, listfile=listfile)
raw = WSDS.RawData(listfile=listfile)
raw.locations = raw.get_locs(mode='latlong')
for ii in range(len(raw.locations)):
    lon, lat = utils.project((raw.locations[ii, 1], raw.locations[ii, 0]), zone=10, letter='U')[2:]
    raw.locations[ii, 1], raw.locations[ii, 0] = lon, lat
data.locations = raw.locations
save_path = 'C:/Users/eroots/phd/ownCloud/Documents/TGI/Figures/'
# rmsites = [site for site in data.site_names if site[0] == 'e' or site[0] == 'd']
# data.remove_sites(rmsites)
# data.sort_sites(order='west-east')
rho = {site.name: utils.compute_rho(site)[0] for site in data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in data.sites.values()}
rho_lim = [0, 90]
n_interp = 250
periods = [7, 12, 21, 38]
padding = 250
# bost = {site.name: utils.compute_bost1D(site)[0] for site in data.sites.values()}
# depths = {site.name: utils.compute_bost1D(site)[1] for site in data.sites.values()}

rho_error_tol = 0.5
phase_error_tol = 10
for idx, period in enumerate(data.periods):
    if idx in periods:
        loc_x = []
        loc_y = []
        rho_vals = []
        phase_vals = []
        loc_z = []
        for dim in (0, 1):
            for ii, site in enumerate(data.site_names):
                # for ii, p in enumerate(data.sites[site].periods):
                    # periods.append(p)
                    # bost2.append(bost[site][ii])
                    # depths2.append(depths[site][ii])
                    phase_tensor = data.sites[site].phase_tensors[idx]
                    # if ((phase_tensor.rhoxy_error / phase_tensor.rhoxy < rho_error_tol) and
                    #     (phase_tensor.rhoyx_error / phase_tensor.rhoyx < rho_error_tol) and
                    #     (phase_tensor.phasexy_error < phase_error_tol) and
                    #     (phase_tensor.phaseyx_error < phase_error_tol)):
                    if not(data.periods[idx] < 4 and site.startswith('98')):
                        rho_vals.append(np.log10(rho[site][idx]))
                        phase_vals.append(pha[site][idx])
                        # # Flip the coords here so X is west-east
                        loc_y.append(raw.locations[ii, 0])
                        loc_x.append(raw.locations[ii, 1])
                        loc_z.append(dim)
        phase_vals = np.array(phase_vals)
        rho_vals = np.array(rho_vals)
        # rho_vals = np.log10(rho_vals)
        loc_x = np.array(loc_x)
        loc_y = np.array(loc_y)
        loc_z = np.array(loc_z)
        points = np.transpose(np.array((loc_x, loc_y, loc_z)))
        # points = np.transpose(np.array((loc_x, loc_y)))

        min_x, max_x = (min(raw.locations[:, 1]) - padding, max(raw.locations[:, 1]) + padding)
        min_y, max_y = (min(raw.locations[:, 0]) - padding, max(raw.locations[:, 0]) + padding)

        grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp),
                                     np.linspace(min_y, max_y, n_interp))
        step_size_x = np.ceil((max_x - min_x) / n_interp)
        step_size_y = np.ceil((max_y - min_y) / n_interp)
        grid_ranges = [[min_x, max_x, step_size_x],
                       [min_y, max_y, step_size_y],
                       [0, 1, 1]]
        grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x, step_size_x),
                                     np.arange(min_y, max_y, step_size_y))
        # grid_rho = griddata(points, rho_vals, (grid_x, grid_y), method='linear')
        # grid_pha = griddata(points, phase_vals, (grid_x, grid_y), method='linear')
        # grid_rho = np.squeeze(nn.griddata(points, rho_vals, grid_ranges))
        grid_pha = np.squeeze(nn.griddata(points, phase_vals, grid_ranges))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        # ax.plot(raw.locations[:, 1] / 1000, raw.locations[:, 0] / 1000, 'k.', markersize=1)
        plt.plot(loc_x / 1000, loc_y / 1000, 'k.', markersize=1)
        im = ax.pcolor(grid_x / 1000, grid_y / 1000, (grid_pha.T), cmap=cmap)
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, cax=cax)
        # cbaxes = fig.add_axes([0.9, -0.212, 0.05, 1.09])
        # cb = plt.colorbar(im)
        ax.set_title('Frequency: {:5.5f} Hz, Period: {:5.5f} s'.format(1 / period, period), fontsize=18)
        ax.set_xlabel('Easting (km)', fontsize=18)
        ax.set_ylabel('Northing (km)', fontsize=18)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=14)
        # cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
        #              rotation=270,
        #              labelpad=30,
        #              fontsize=14)
        im.set_clim(rho_lim)
        # plt.gca().invert_yaxis()
        plt.savefig(save_path + 'pha_' + str(idx) + '.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(save_path + 'pha_' + str(idx) + '.png', dpi=600, bbox_inches='tight')
        # plt.close()
        # plt.figure(figsize=(8, 8))
        # ax = plt.gca()
        # plt.plot(loc_x / 1000, loc_y / 1000, 'k.', markersize=1)
        # im = plt.pcolor(grid_x / 1000, grid_y / 1000, grid_pha.T, cmap=cmap)
        # ax.set_title('Period: {:5.5f}'.format(period))
        # ax.set_xlabel('Easting (km)')
        # ax.set_ylabel('Northing (km)')
        # ax.set_aspect('equal')
        # # divider = make_axes_locatable(ax)
        # # cax = divider.append_axes("right", size="5%", pad=0.05)
        # # cb = plt.colorbar(im, cax=cax)
        # # cb.set_label(r'Phase ($^\circ$)',
        # #              rotation=270,
        # #              labelpad=30,
        # #              fontsize=14)
        # plt.clim([0, 90])
        # plt.savefig(save_path + 'pha_' + str(idx) + '.pdf', dpi=600)
        # plt.savefig(save_path + 'pha_' + str(idx) + '.png', dpi=600)
        # # plt.gca().invert_yaxis()
        # plt.close()
        # plt.show()
