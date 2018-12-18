import shapefile
import numpy as np
import matplotlib.pyplot as plt
import pyMT.data_structures as WSDS
import pyMT.utils as utils
import cartopy.crs as ccrs
from cartopy.io import shapereader
import matplotlib
# from cartopy.feature import ShapelyFeature
# from scipy.interpolate import griddata
import naturalneighbor as nn
import e_colours.colourmaps as cm
# from mpl_toolkits.axes_grid1 import make_axes_locatable


shp_file_base = 'C:/Users/eric/phd/ownCloud/data/ArcMap/test2.shp'
list_file = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/culled_allSuperior.lst'
datafile = 'C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/j2/cull_allSuperior.data'
cmap = cm.jet_plus(64)
rho_lim = [0, 5]
n_interp = 250
period = 14
padding = 50000
rho_error_tol = 0.5
phase_error_tol = 10
save_fig = 1

save_path = 'C:/Users/eric/phd/ownCloud/Documents/Seminars/Seminar 3/Figures/Pseudosections/subprovinces/botCBar/'

# raw = WSDS.RawData(list_file)
# data = WSDS.Data(datafile=datafile, listfile=list_file)
use_periods = data.periods
raw.locations = raw.get_locs(mode='latlong')
# transform = ccrs.PlateCarree()
# We want the data plotted in UTM, and we will convert them to UTM before plotting
# transform = ccrs.UTM(zone=16)
transform = ccrs.TransverseMercator(central_longitude=-85, central_latitude=49,
                                    false_northing=5430000, false_easting=645000)
for ii in range(len(raw.locations)):
    easting, northing = utils.project((raw.locations[ii, 1], raw.locations[ii, 0]), zone=16, letter='U')[2:]
    raw.locations[ii, 1], raw.locations[ii, 0] = easting, northing



shp = shapereader.Reader(shp_file_base)
# Note I use ccrs.PlateCarree() here because that is the projection the shapefile is in
# I.E., latlong, not UTM. cartopy will take care of converting them as long as these are
# all defined properly.
# plt.plot(raw.locations[:, 1], raw.locations[:, 0], 'k.', transform=transform)
# plt.show()

data.locations = raw.locations
rho = {site.name: utils.compute_rho(site)[0] for site in data.sites.values()}
pha = {site.name: utils.compute_phase(site)[0] for site in data.sites.values()}
for idx, period in enumerate(use_periods):
    loc_x = []
    loc_y = []
    rho_vals = []
    phase_vals = []
    loc_z = []
    for dim in (0, 1):
        for ii, site in enumerate(data.site_names):
                phase_tensor = data.sites[site].phase_tensors[idx]
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
    grid_rho = np.squeeze(nn.griddata(points, rho_vals, grid_ranges))
    # grid_pha = np.squeeze(nn.griddata(points, phase_vals, grid_ranges))
    
    # ax = fig.add_subplot(111)
    # ax.plot(raw.locations[:, 1] / 1000, raw.locations[:, 0] / 1000, 'k.', markersize=1)
    # This defines the projection we want to plot in
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=transform)
    ax.coastlines()
    for record, shape in zip(shp.records(), shp.geometries()):
        ax.add_geometries([shape], ccrs.PlateCarree(), facecolor='none', edgecolor='black')
    # shape_feature = ShapelyFeature(Reader(shp_file_base).geometries(),
    #                                transform, edgecolor='black')
    # ax.add_feature(shape_feature, facecolor='blue')
    # plt.show()
    for val, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        label.set_text(str(val))
        label.set_position((val, 0))

    for val, label in zip(ax.get_yticks(), ax.get_yticklabels()):
        label.set_text(str(val))
        label.set_position((0, val))

    plt.tick_params(bottom=True, top=True, left=True, right=True,
                    labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    plt.grid(True)
    plt.plot(loc_x, loc_y, 'w.', markersize=1, transform=transform)
    im = plt.pcolor(grid_x, grid_y, (grid_rho.T), cmap=cmap, transform=transform)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    # cbaxes = fig.add_axes([0.9, -0.212, 0.05, 1.09])
    # cb = plt.colorbar(im)
    ax.set_title('Frequency: {:5.5g} Hz, Period: {:5.5g} s'.format(1 / period, period), fontsize=18)
    ax.set_xlabel('Easting (m)', fontsize=18)
    ax.set_ylabel('Northing (m)', fontsize=18)
    # ax.axes().set_aspect('equal')
    ax.tick_params(axis='both', labelsize=14)
    im.set_clim(rho_lim)
    cax, kw = matplotlib.colorbar.make_axes(ax,
                                            location='bottom',
                                            pad=0.125,
                                            shrink=0.9,
                                            extend='both')
    cb = fig.colorbar(im, cax=cax, **kw)
    cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
                 rotation=0,
                 labelpad=10,
                 fontsize=14)
    if save_fig:
        plt.savefig(save_path + 'rho_' + str(idx) + '.pdf', dpi=600, bbox_inches='tight')
        plt.savefig(save_path + 'rho_' + str(idx) + '.png', dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
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
