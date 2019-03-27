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
import e_colours.colourmaps


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
    res_vals = 1 / resolution.vals
    res_vals = res_vals / volumes ** (1 / 3)
    res_vals[res_vals > np.mean(res_vals.flatten())] = np.mean(res_vals.flatten())
    res_vals = res_vals - np.mean(res_vals.flatten())
    res_vals = res_vals / np.std(res_vals.flatten())
    res_vals = 0.5 + res_vals * np.sqrt(0.5)
    res_vals[res_vals > 1] = 1
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


# mod = WSDS.Model(r'C:\Users\eric\Documents' +
#                  r'\MATLAB\MATLAB\Inversion\Regions' +
#                  r'\abi-gren\New\abi0_sens\outSens_model.00')
# reso = WSDS.Model(r'C:\Users\eric\Documents' +
#                   r'\MATLAB\MATLAB\Inversion\Regions' +
#                   r'\abi-gren\New\abi0_sens\Resolution0_inverted.model')
# data = WSDS.RawData(r'C:\Users\eric\Documents' +
#                     r'\MATLAB\MATLAB\Inversion\Regions' +
#                     r'\abi-gren\New\j2\test.lst',
#                     datpath=r'C:\Users\eric\Documents' +
#                     r'\MATLAB\MATLAB\Inversion\Regions' +
#                     r'\abi-gren\New\j2')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\gem_thelon\original\sensTest2.model')
# data = WSDS.RawData(listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\gem_thelon\original\all_sites.lst',
#                     datpath=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\j2')
# mod = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\abi-gren\center_ModEM\NLCG\center_noTF_final.model'))
# data = WSDS.RawData(listfile=r'C:\Users\eroots\phd\ownCloud\data\Regions\abi-gren\j2\center_fewer3.lst')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_bost1\out_model.00')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_3\sens_model.00')
# reso = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_3\Resolution_inverted.model')
# data = WSDS.RawData(listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\gem_thelon\original\all_sites.lst',
#                     datpath=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\j2')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Occam\OCCAM2DMT_V3.0\dbrSlantedFaults\faulted_v8L\dbr_occUVT_Left.model'))
# data = WSDS.RawData(listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsitesBBMT.lst'))
# data = WSDS.RawData(listfile='C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/cull5.lst')
# mod = WSDS.Model('C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/wst0_sens/wst0Inv5_model.02')

# data = WSDS.RawData(listfile='C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/southcentral2.lst')
# mod = WSDS.Model('C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/wsSC1/wsSC_final.model')
# data = WSDS.RawData(listfile='C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/southeastern_2.lst')
# mod = WSDS.Model('C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/wsSE3_ModEM/wsSE3TF_final.model')
# seismic = pd.read_table('C:/Users/eric/Desktop/andy/WS1-cdp.dat.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')

# data = WSDS.RawData(listfile='F:/ownCloud/data/Regions/wst/j2/southeastern_2.lst')
# mod = WSDS.Model('F:/ownCloud/data/Regions/wst/wsSE3_ModEM/wsSE3TF_final.model')
# data = WSDS.RawData(listfile='F:/ownCloud/data/Regions/wst/j2/southcentral.lst')
# mod = WSDS.Model('F:/ownCloud/data/Regions/wst/wsSC1/finish/wsSC_finish.model')
# mod = WSDS.Model('F:/ownCloud/data/Regions/wst/wsSC1/wsSC_final.model')
# data = WSDS.RawData(listfile='C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/southeastern_2.lst')
# mod = WSDS.Model('C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/wsSE3_ModEM/wsSE3TF_final.model')
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst')
# used_data = WSDS.Data(datafile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_cull1i.dat',
#                  listfile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_finish.model')
#########################################################
# SWAYZE
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_finish.model')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/pt/swzPT3_lastIter.rho')
# mod.vals = np.log10(mod.vals) - np.log10(mod2.vals)
#########################################################
# DRYDEN-ATIKOKAN
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry53.rho')
#########################################################
# seismic = pd.read_table('F:/ownCloud/andy/navout_600m.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
# qx, qy = (np.array(seismic['x'] / 1000),
          # np.array(seismic['y']) / 1000)
# data.locations = data.get_locs(site_list=main_transect.site_names)
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect_north.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R1North_cull2.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/R1North_1/finish/finish2_lastIter.rho')
##########################################################
# MALARTIC
main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect.lst')
data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/mal3_lastIter.rho')
seismic = pd.read_table(r'C:\Users\eroots\Downloads\Malartic Seismic Receivers location (1)\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
use_seismic = 1
# azi = 35  # Dryden-regional
# azi = -15  # Swayze regional
azi = 0  # Malartic regional
# UTM_number = 16
# UTM_letter = 'U'
UTM_number = 17
UTM_letter = 'N'
# padding = 25000
padding = 10000
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 3
file_path = r'C:/Users/eroots/phd/ownCloud/Documents/Swayze_paper/RoughFigures/'
file_name = 'swzRegional_0-5_PT-Z_logScale'
file_types = ['.pdf', '.ps', '.png']
title_ = 'Standard Inversion'

save_fig = 0
save_dat = 0
dpi = 600
csv_name = 'C:/Users/eroots/phd/ownCloud/Metal Earth/Data/model_csvs/swayze_regional.dat'
use_alpha = 0
saturation = 0.8
lightness = 0.4

xlim = []
zlim = [0, 75]
# zlim = [0, 400]
lut = 64
isolum = False
# xlim = [-123.5, -121.5]
# xlim = [-7, 74]
# zlim = [0, 5]
lut = 256
cax = [0, 4.5]
# cax = [-2, 2]
isolum = 0
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
# cmap_name = 'gray'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
cmap_name = 'jetplus'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'
# cmap_name = 'jetplus'

data.locations = data.get_locs(mode='latlong')
for ii in range(len(data.locations)):
        easting, northing = utils.project((data.locations[ii, 1],
                                           data.locations[ii, 0]),
                                          zone=UTM_number, letter=UTM_letter)[2:]
        data.locations[ii, 1], data.locations[ii, 0] = easting, northing
data.locations = utils.rotate_locs(data.locations, azi)
origin = data.origin
mod.origin = origin
# seismic = pd.read_table('F:/ownCloud/andy/navout_600m.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
# if seismic:
#     qx, qy = (np.array(seismic['x'] / 1000),
#               np.array(seismic['y']) / 1000)
# else:
#     qx, qy = [], []
#     site_x, site_y = [main_transect.locations[:, 1] / 1000,
#                       main_transect.locations[:, 0] / 1000]

#     for ii in range(len(site_x) - 1):
#         qx.append(np.linspace(site_x[ii], site_x[ii + 1], 100).ravel())
#         qy.append(np.linspace(site_y[ii], site_y[ii + 1], 100).ravel())
#     qx = np.array(qx).ravel()
#     qy = np.array(qy).ravel()

reso = []
kimberlines = []
mod.origin = data.origin
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
data.locations = data.locations[data.locations[:, 0].argsort()]  # Make sure they go north-south
# A little kludge to make sure the last few sites are in the right order (west-east)
# data.locations[1:8, :] = data.locations[np.flip(data.locations[1:8, 1].argsort())]
if use_seismic:
        qx, qy = (np.array(seismic['x'] / 1000),
                  np.array(seismic['y']) / 1000)
else:
    X = np.linspace(data.locations[0, 0] - padding, data.locations[0, 0], 20)
    Y = np.interp(X, data.locations[:, 0], data.locations[:, 1])
    qx = []
    qy = []
    qx.append(Y)
    qy.append(X)
    for ii in range(len(data.locations[:, 0]) - 1):
        X = np.linspace(data.locations[ii, 0], data.locations[ii + 1, 0], 50)
        Y = np.interp(X, data.locations[:, 0], data.locations[:, 1])
        qx.append(Y)
        qy.append(X)
    X = np.linspace(data.locations[-1, 0], data.locations[-1, 0] + padding, 20)
    Y = np.interp(X, data.locations[:, 0], data.locations[:, 1])
    qx.append(Y)
    qy.append(X)
    qx = np.concatenate(qx).ravel() / 1000
    qy = np.concatenate(qy).ravel() / 1000
reso = []
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
if cmap_name == 'jetplus':
    cmap = e_colours.colourmaps.jet_plus(lut)
else:
    cmap = cm.get_cmap(cmap_name, lut)
qz = []
for ii in range(len(z) - 1):
    qz.append(np.linspace(z[ii], z[ii + 1], 5))
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
    alpha = alpha[:, 73, :]
# if mode == 2:
    # vals = interpolate_slice(y, z, vals, 300)
print('Interpolating...')
# vals = griddata(data_points, data_values, query_points, 'nearest')
interpolator = RGI((y, x, z), np.transpose(vals, [1, 0, 2]), bounds_error=False, fill_value=5)
vals = interpolator(query_points)
vals = np.reshape(vals, [len(qx), len(qz)])
if reso:
    alpha = interpolate_slice(y, z, alpha, 300)

norm_vals = (vals - min(cax[0], np.min(vals))) / \
            (max(np.max(vals), cax[1]) - min(cax[0], np.min(vals)))
if mode == 2:
    rgb = cmap(np.flipud(norm_vals.T))
    if reso:
        alpha = np.flipud(alpha.T)
else:
    rgb = cmap(norm_vals.T)
    if reso:
        alpha = alpha.T
# Just turn the bottom row transparent, since those cells often somehow still have resolution
# alpha[-1, :] = alpha[-1, :].clip(max=0.5)
rgba = copy.deepcopy(rgb)
if reso:
    rgba[..., -1] = alpha



# Rotate locations back to true coordinates
if azi:
    data.locations = utils.rotate_locs(data.locations, azi=-azi)
    p = utils.rotate_locs(np.array((qx, qy)).T, azi=azi)
    qx = p[:, 0]
    qy = p[:, 1]
# cmap[..., -1] = reso.vals[:, 31, :]

# I had to change the way things plotted, so isolum is unusable right now.
# Issue is: if we are normalizing data to fit between 0 and 1 so that we can
# convert it to a cmap, that assumes that we actually have values at the max
# and min of the cbar we want to use. So, e.g., if we have data from 15-100000, and try
# to use a cbar from log10(10)) to log10((100000)), the min value of 15 will get mapped
# down to 10.
if isolum and reso:
    cmap = cmap(np.arange(lut))
    cmap = cmap[:, :3]
    rgb = rgba[:, :, :3]

    cmap = rgb2hls(cmap)
    cmap[:, 2] = saturation
    cmap[:, 1] = lightness
    cmap = hls2rgb(cmap)
    hls = rgb2hls(rgb)
    hls[:, :, 2] = saturation
    hls[:, :, 1] = lightness
    rgb = hls2rgb(hls)
    rgba = np.dstack([rgb, alpha.clip(min=0.1, max=1)])
    hls[:, :, 1] = alpha.clip(min=0.3, max=lightness)
    rgba_l = hls2rgb(hls)
    hls[:, :, 1] = np.abs(1 - alpha).clip(min=lightness, max=0.7)
    rgba_lr = hls2rgb(hls)

    # value = 0.5
    # saturation = 0.5
    # cmap = colors.rgb_to_hsv(cmap)
    # cmap[:, 1] = saturation
    # cmap[:, 2] = value
    # cmap = colors.hsv_to_rgb(cmap)
    # hsv = colors.rgb_to_hsv(rgb)
    # hsv[:, :, 1] = saturation
    # hsv[:, :, 2] = value
    # rgb = colors.hsv_to_rgb(hsv)
    # # if use_alpha:
    # rgba = colors.hsv_to_rgb(hsv)
    # rgba = np.dstack([rgba, alpha])
    # # else:
    # hsv[:, :, 2] = alpha.clip(min=0.2, max=value)
    # rgba_l = colors.hsv_to_rgb(hsv)
    # alpha_l = np.abs(1 - alpha).clip(min=value, max=1)
    # hsv[:, :, 2] = alpha_l
    # rgba_lr = colors.hsv_to_rgb(hsv)

    cmap = colors.ListedColormap(cmap, name='test1')

fig = plt.figure(1, figsize=(8, 4.5))
h = [Size.Fixed(0.), Size.Fixed(6.5)]
v = [Size.Fixed(0.5), Size.Fixed(3.25)]
win = Divider(fig, (0.1, 0.1, 0.8, 0.8), h, v, aspect=False)
ax = Axes(fig, win.get_position())
ax.set_axes_locator(win.new_locator(nx=1, ny=1))
fig.add_axes(ax)
# fig = plt.gcf()
for ii in range(1, 2):
    if ii == 0:
        to_plot = rgb
        title = 'Model'
    elif ii == 1:
        to_plot = rgba
        title = 'Model + Resolution (Transparency)'
    elif ii == 2:
        to_plot = rgba_l
        title = 'Model + Resolution (Lightness)'
    else:
        to_plot = rgba_lr
        title = 'Model + Resolution (Reverse Lightness)'
    # ax = fig.add_subplot(2, 2, ii + 1)
    # ax = fig.add_subplot(1, 1, 1)
    if mode == 1:
        # Have to kind of hack this a bit
        # Cut the first and last cell in half to squeeze the model into the same
        # space as for the 'imshow' case.
        mod.dy[0] = (mod.dy[0] + mod.dy[1]) / 2
        mod.dy[-1] = (mod.dy[-1] + mod.dy[-2]) / 2
        im = plt.pcolormesh(np.array(mod.dy) / 1000, np.array(mod.dz) / 1000, vals.T,
                            vmin=1, vmax=5, cmap=cmap, edgecolor='k', linewidth=0.01)
    elif mode == 2:
        im = plt.imshow(np.flipud(vals.T), cmap=cmap, vmin=np.log10(np.min(vals)),
                        vmax=np.log10(np.max(vals)), aspect='auto',
                        extent=[y[0], y[-1], z[0] / 1000, z[-1] / 1000])
    elif mode == 3:
        # mod.dx[0] = (mod.dx[0] + mod.dx[1]) / 2
        # mod.dx[-1] = (mod.dx[-1] + mod.dx[-2]) / 2
        to_plot = to_plot[1:, 1:]
        im, ax = pcolorimage(ax,
                             x=(np.array(qy)),
                             y=np.array(qz),
                             A=(to_plot), cmap=cmap)
        # sites = ax.plot(data.locations[:, 0] / 1000,
        #                 np.zeros(len(data.locations[:, 1])) - 0.5,
        #                 'wv', mec='k', markersize=7)
        # ax.set_aspect('equal')
        # sites[0].set_clip_on(False)
    if xlim:
        ax.set_xlim(xlim)
    if zlim:
        ax.set_ylim(zlim)
    ax.invert_yaxis()
    # ax.invert_xaxis()
    # ax.set_xlabel('Latitude', fontsize=20)
    # ax.set_ylabel('Depth (km)', fontsize=20)
    # ax.set_title(title_, y=1.02, fontsize=20)
    fig.canvas.draw()
    # if mod.coord_system == 'latlong':
    #     labels = [item.get_text() for item in ax.get_xticklabels()]
    #     for jj, item in enumerate(labels):
    #         if item:
    #             labels[jj] = ''.join([item, '$^\circ$N'])
    #     ax.set_xlabel('Longitude', fontsize=20)
    #     ax.set_xticklabels(labels)
    # elif mod.coord_system == 'UTM':
    #     ax.set_xlabel('Northing', fontsize=20)
    # else:
    #     ax.set_xlabel('X (km)', fontsize=20)
    # for line in kimberlines:
    #     ax.plot([line, line], [0, 200], 'w-', lw=0.5)

ax.autoscale_view(tight=True)
ax.tick_params(axis='both', labelsize=14)
locs = ax.plot(data.locations[:, 0] / 1000,
               np.zeros((data.locations.shape[0])) - 0.5,
               'kv', markersize=6)[0]
for jj, site in enumerate(data.site_names):
    plt.text(s=site,
             x=data.locations[jj, 0] / 1000,
             y=-7.5,
             color='k',
             rotation=90)
locs.set_clip_on(False)
# for label in ax.xaxis.get_ticklabels():
#     label.set_visible(False)
# for label in ax.yaxis.get_ticklabels():
#     label.set_visible(False)
# ax.tick_params(axis='y', labelsize=10)
# fig.subplots_adjust(right=0.8)

divider = make_axes_locatable(ax)
cb_ax = divider.append_axes('right', size='2.5%', pad=0.1)
cb = plt.colorbar(im, cmap=cmap, cax=cb_ax, orientation='vertical', extend='both')
cb.set_clim(cax[0], cax[1])
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
             rotation=270,
             labelpad=30,
             fontsize=14)
cb.draw_all()

# figlegend = plt.figure(figsize=(4, 4))
# figlegend.legend(sites, ('Site Locations', ''), 'center')
# figlegend.show()
# figlegend.savefig('legend.pdf')
# fig.set_dpi(dpi)
# ax.set_aspect(3)
if save_fig:
    for ext in file_types:
        fig.savefig(file_path + file_name + ext, dpi=dpi,
                    transparent=True)

if save_dat:
    x_loc = np.tile(1000 * qx[:, np.newaxis], [vals.shape[-1]]).ravel()
    y_loc = np.tile(1000 * qy[:, np.newaxis], [vals.shape[-1]]).ravel()
    z_loc = np.tile(1000 * qz, len(qx))
    # cdp = np.array(seismic['cdp'])
    cdp = np.array(range(1, len(qx) + 1))
    cdp = np.tile(cdp[:, np.newaxis], [vals.shape[-1]]).ravel()
    df = pd.DataFrame(np.array((cdp, x_loc, y_loc, z_loc, np.ravel(vals))).T, columns=None)
    df.to_csv(''.join([csv_name, 'log10.dat']), sep=',', header=None, index=False)
    df = pd.DataFrame(np.array((cdp, x_loc, y_loc, z_loc, 10 ** (np.ravel(vals)))).T, columns=None)
    df.to_csv(''.join([csv_name, 'linear.dat']), sep=',', header=None, index=False)
plt.show()
