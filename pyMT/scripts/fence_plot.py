import pyMT.data_structures as WSDS
import pyMT.utils as utils
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.image import PcolorImage
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
import naturalneighbor as nn
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
# from scipy.interpolate import SmoothBivariateSpline as RBS
import copy
# import colorcet as cc
import colorsys
from pyMT.e_colours import colourmaps


local_path = 'C:/Users/eroots'
# local_path = 'C:/Users/eric/'


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[1] + delta / 2]


def rgb2hls(rgb):
    hls = np.zeros(rgb.shape)
    if rgb.ndim == 3:
        for ix in range(rgb.shape[0]):
            for iy in range(rgb.shape[1]):
                hls[ix, iy, :] = colorsys.rgb_to_hls(rgb[ix, iy, 0],
                                                     rgb[ix, iy, 1],
                                                     rgb[ix, iy, 2])
    else:
        for ix in range(rgb.shape[0]):
                hls[ix, :] = colorsys.rgb_to_hls(rgb[ix, 0],
                                                 rgb[ix, 1],
                                                 rgb[ix, 2])
    return hls


def hls2rgb(hls):
    rgb = np.zeros(hls.shape)
    if rgb.ndim == 3:
        for ix in range(hls.shape[0]):
            for iy in range(hls.shape[1]):
                rgb[ix, iy, :] = colorsys.hls_to_rgb(hls[ix, iy, 0],
                                                     hls[ix, iy, 1],
                                                     hls[ix, iy, 2])
    else:
        for ix in range(rgb.shape[0]):
            rgb[ix, :] = colorsys.hls_to_rgb(hls[ix, 0],
                                             hls[ix, 1],
                                             hls[ix, 2])
    return rgb


def normalize_resolution(model, resolution):
    xCS, yCS, zCS = [np.diff(mod.dx),
                     np.diff(mod.dy),
                     np.diff(mod.dz)]
    X, Y, Z = np.meshgrid(yCS, xCS, zCS)
    volumes = X * Y * Z
    res_vals = resolution.vals
    res_vals = res_vals / volumes ** (1 / 3)
    res_vals[res_vals > np.mean(res_vals.flatten())] = np.mean(res_vals.flatten())
    res_vals = res_vals - np.mean(res_vals.flatten())
    res_vals = res_vals / np.std(res_vals.flatten())
    res_vals = 0.2 + res_vals * np.sqrt(0.2)
    res_vals = np.clip(res_vals, a_min=0, a_max=1)
    return res_vals


def pcolorimage(ax, x=None, y=None, A=None, **kwargs):
    img = PcolorImage(ax, x, y, A, **kwargs)
    img.set_extent([x[0], x[-1], y[0], y[-1]])  # Have to add this or the fig won't save...
    ax.images.append(img)
    ax.set_xlim(left=x[0], right=x[-1])
    ax.set_ylim(bottom=y[0], top=y[-1])
    # ax.autoscale_view(tight=True)
    ax.set_xlabel('Northing (km)', fontsize=14)
    ax.set_ylabel('Depth (km)', fontsize=14)
    return img, ax


def interpolate_slice(x, y, Z, NP):
    mod_interp = RBS(x, y, Z)
    interp_vals = mod_interp(np.linspace(x[0], x[-1], NP),
                             np.linspace(y[0], y[-1], NP))
    return interp_vals


def project_locations(data, zone, letter):
    data.locations = data.get_locs(mode='latlong')
    for ii in range(len(data.locations)):
        easting, northing = utils.project((data.locations[ii, 1],
                                           data.locations[ii, 0]),
                                          zone=zone, letter=letter)[2:]
        data.locations[ii, 1], data.locations[ii, 0] = easting, northing
    return data

#########################################################
# AFTON
main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/l3.lst')
data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_aroundOre.lst')
backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_aroundOre.lst')
mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton2/norot/afton2_lastIter.rho')

use_seismic = 0
use_trace_ticks = 0
force_NS = 1

azi = 0  # Malartic regional
reso = []
ninterp = 20
padding = 1000
ninterp_padding = 100
mode = 3
file_path = local_path + '/phd/ownCloud/Documents/ME_Transects/Matheson/Haugaard2019/Figures/HexSlices/12km/'
file_name = 'Matheson_east_HexMT_all_turbo0-5'
file_types = ['.png']
title_ = 'Standard Inversion'
rotate_back = 1
linear_xaxis = True
save_fig = 0
save_dat = 0
annotate_sites = 0
site_markers = 1
# site_markers = 0
plot_map = 1
dpi = 600
csv_name = 'C:/Users/eroots/phd/ownCloud/Metal Earth/Data/model_csvs/swayze_regional.dat'
use_alpha = 0
saturation = 0.8
lightness = 0.4
xlim = []
zlim = [0, 2]
# zlim = [0, 6]
# zlim = [0, 30]
lut = 32
# zlim = [0, 100]
# lut = 64
isolum = False
cax = [0, 3]
isolum = 0
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
cmap_name = 'turbo_r'
# cmap_name = 'gray'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'
# cmap_name = 'jetplus'
nudge_sites = main_transect.site_names
# reverse_nudge = ['MAL008M', 'MAL009M', 'MAL010L']
reverse_nudge = []

# nudge_dist = 5000
nudge_dist = -750
use_nudge = 0

# # Make sure the sites go north-south
# if force_NS:
main_transect.locations = main_transect.locations[main_transect.locations[:, 0].argsort()]
# Sort the site names so the same is true
main_transect.site_names = sorted(main_transect.site_names,
                                  key=lambda x: main_transect.sites[x].locations['X'])
nudge_locations = copy.deepcopy(main_transect.locations)
if use_nudge:
    for ii, site in enumerate(main_transect.site_names):
        if site in nudge_sites:
            nudge_locations[ii, 1] += nudge_dist
        elif site in reverse_nudge:
            nudge_locations[ii, 1] -= nudge_dist
if plot_map and not use_seismic:
    site_x, site_y = [main_transect.locations[:, 1],
                      main_transect.locations[:, 0]]
    qx_map, qy_map = [], []
    X = np.linspace(nudge_locations[0, 0] - padding, nudge_locations[0, 0], ninterp_padding)
    Y = np.interp(X, site_y, site_x)
    qx_map.append(Y)
    qy_map.append(X)
    # for ii in range(len(site_x) - 1):
    for ii, site in enumerate(main_transect.site_names[:-1]):
        # if use_nudge:
        use_x, use_y = nudge_locations[:, 1], nudge_locations[:, 0]
        # else:
            # use_x, use_y = site_x, site_y
        X = np.linspace(use_y[ii], use_y[ii + 1], ninterp)
        Y = np.interp(X, use_y, use_x)
        qx_map.append(Y)
        qy_map.append(X)
    X = np.linspace(use_y[-1], use_y[-1] + padding, ninterp_padding)
    Y = np.interp(X, use_y, use_x)
    qx_map.append(Y)
    qy_map.append(X)
    qx_map = np.concatenate(qx_map).ravel()
    qy_map = np.concatenate(qy_map).ravel()

data = copy.deepcopy(main_transect)
data.locations = utils.rotate_locs(data.locations, azi)
origin = backup_data.origin  # Has to be this since 'data' has been modified and so is no longer in sync with model
# mod.origin = origin
mod.origin = origin
mod.to_UTM()
if mod.coord_system == 'UTM':
    mod.dx = [xx / 1000 for xx in mod.dx]
    mod.dy = [yy / 1000 for yy in mod.dy]
    mod.dz = [zz / 1000 for zz in mod.dz]

idx = []
rm_sites = []
for ii, site in enumerate(data.site_names):
    if site not in main_transect.site_names:
        idx.append(ii)
        rm_sites.append(site)
data.locations = np.delete(data.locations, idx, axis=0)
data.site_names = [site for site in data.site_names if site not in rm_sites]
if force_NS:
    data.locations = data.locations[data.locations[:, 0].argsort()]  # Make sure they go north-south
# A little kludge to make sure the last few sites are in the right order (west-east)
# data.locations[1:8, :] = data.locations[np.flip(data.locations[1:8, 1].argsort())]
# nudge_locations = copy.deepcopy(data.locations)
# for ii, site in enumerate(data.site_names):
#     if site in nudge_sites:
#         nudge_locations[ii, 1] += nudge_dist
if use_seismic:
        qx, qy = (np.array(seismic['x'] / 1000),
                  np.array(seismic['y']) / 1000)
        if azi:
            locs = utils.rotate_locs(np.array((qy, qx)).T, azi)
            qx, qy = locs[:, 1], locs[:, 0]
        if plot_map:
            qx_map = copy.deepcopy(qx) * 1000
            qy_map = copy.deepcopy(qy) * 1000
else:
    X = np.linspace(nudge_locations[0, 0] - padding, nudge_locations[0, 0], ninterp_padding)
    Y = np.interp(X, nudge_locations[:, 0], nudge_locations[:, 1])
    qx = []
    qy = []
    qx.append(Y)
    qy.append(X)
    # for ii in range(len(data.locations[:, 0]) - 1):
    for ii, site in enumerate(data.site_names[:-1]):
        if use_nudge:
            use_x, use_y = nudge_locations[:, 1], nudge_locations[:, 0]
        else:
            use_x, use_y = data.locations[:, 1], data.locations[:, 0]
        X = np.linspace(use_y[ii], use_y[ii + 1], ninterp)
        Y = np.interp(X, use_y[:], use_x[:])
        qx.append(Y)
        qy.append(X)
    X = np.linspace(use_y[-1], use_y[-1] + padding, ninterp_padding)
    Y = np.interp(X, use_y[:], use_x[:])
    qx.append(Y)
    qy.append(X)
    qy_2D = np.concatenate(qy) / 1000
    qx_2D = np.concatenate(qx) / 1000
    qx = np.concatenate(qx).ravel() / 1000
    qy = np.concatenate(qy).ravel() / 1000
kimberlines = []

x, y, z = [np.zeros((len(mod.dx) - 1)),
           np.zeros((len(mod.dy) - 1)),
           np.zeros((len(mod.dz) - 1))]
for ii in range(len(mod.dx) - 1):
    x[ii] = (mod.dx[ii] + mod.dx[ii + 1]) / 2
for ii in range(len(mod.dy) - 1):
    y[ii] = (mod.dy[ii] + mod.dy[ii + 1]) / 2
for ii in range(len(mod.dz) - 1):
    z[ii] = (mod.dz[ii] + mod.dz[ii + 1]) / 2
if cmap_name in ('jetplus', 'turbo', 'turbo_r'):
    cmap = colourmaps.get_cmap(cmap_name, lut)
else:
    cmap = cm.get_cmap(cmap_name, lut)
qz = []
for ii in range(len(z) - 1):
    qz.append(np.linspace(z[ii], z[ii + 1], 5))
qz_2D = np.concatenate(qz) / 1000
qz = np.array(qz).ravel()

#  Important step. Since we are normalizing values to fit into the colour map,
#  we first have to threshold to make sure our colourbar later will make sense.
vals = np.log10(mod.vals)
vals[vals < cax[0]] = cax[0]
vals[vals > cax[1]] = cax[1]

#  Build Mx3 array of data points
data_points = np.zeros((mod.vals.size, 3))
data_values = np.zeros((mod.vals.size))
print('Number of data points: {}'.format(data_values.size))
cc = 0
for ix in range(len(x)):
    for iy in range(len(y)):
        for iz in range(len(z)):
            data_values[cc] = vals[ix, iy, iz]
            data_points[cc, :] = np.array((y[iy], x[ix], z[iz]))
            cc += 1

query_points = np.zeros((len(qx) * len(qz), 3))
#  Build Nx3 array of query points
cc = 0
print('Number of query points: {}'.format(query_points.size))
for ix in range(len(qx)):
        for iz in qz:
            query_points[cc, :] = np.array((qx[ix], qy[ix], iz))
            cc += 1

if reso:
    alpha = normalize_resolution(mod, reso)
    # alpha = alpha[11, :, :]
# if mode == 2:
    # vals = interpolate_slice(y, z, vals, 300)
print('Interpolating...')
# vals = griddata(data_points, data_values, query_points, 'nearest')
interpolator = RGI((y, x, z), np.transpose(vals, [1, 0, 2]), bounds_error=False, fill_value=5)
vals = interpolator(query_points)
vals = np.reshape(vals, [len(qx), len(qz)])
if reso:
    # alpha = interpolate_slice(y, z, alpha, 300)
    interpolator_alpha = RGI((y, x, z), np.transpose(alpha, [1, 0, 2]), bounds_error=False, fill_value=0)
    alpha = interpolator_alpha(query_points)
    alpha = np.reshape(alpha, [len(qx), len(qz)])
    alpha = np.clip(alpha, a_max=0.99, a_min=0)

norm_vals = (vals - min(cax[0], np.min(vals))) / \
            (max(np.max(vals), cax[1]) - min(cax[0], np.min(vals)))

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(ax,
                x=qy_2D,
                y=qx_2D,
                z=qz_2D)