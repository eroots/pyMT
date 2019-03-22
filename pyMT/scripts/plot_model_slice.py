import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.image import PcolorImage
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
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
    res_vals = resolution.vals
    res_vals = res_vals / volumes ** (1 / 3)
    res_vals[res_vals > np.median(res_vals.flatten())] = np.median(res_vals.flatten())
    res_vals = res_vals - np.mean(res_vals.flatten())
    res_vals = res_vals / np.std(res_vals.flatten())
    res_vals = 0.5 + res_vals * np.sqrt(0.5)
    res_vals = np.clip(res_vals, a_max=1, a_min=0)
    return res_vals


def pcolorimage(ax, x=None, y=None, A=None, **kwargs):
    img = PcolorImage(ax, x, y, A, **kwargs)
    img.set_extent([x[0], x[-1], y[0], y[-1]])  # Have to add this or the fig won't save...
    ax.images.append(img)
    ax.set_xlim(left=x[0], right=x[-1])
    ax.set_ylim(bottom=y[0], top=y[-1])
    ax.autoscale_view(tight=True)
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
# mod = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\abi-gren\center_ModEM\NLCG\center_noTF_final.model')
# data = WSDS.RawData(listfile=r'C:\Users\eroots\phd\ownCloud\data\Regions\abi-gren\j2\center_fewer3.lst')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_bost1\out_model.00')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_3\sens_model.00')
# reso = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\ttz0_3\Resolution_inverted.model')
# data = WSDS.RawData(listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\gem_thelon\original\all_sites.lst',
#                     datpath=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\TTZ\j2')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Occam\OCCAM2DMT_V3.0\dbrSlantedFaults\faulted_v8L\dbr_occUVT_Left.model')
# data = WSDS.RawData(listfile=r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\dbr15\j2\allsitesBBMT.lst')
mod = WSDS.Model(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.model')
data = WSDS.Data(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_cull1i_Z.dat')
reso = WSDS.Model(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\Resolution.model')
# mod = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\Resolution.model')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2e_smooth.model')
# data = WSDS.Data(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2f_bb_Z.dat')
# data = WSDS.Data(datafile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry53.data',
#                  listfile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry53.rho')
# kimberlines = [5.341140e+006, 5.348097e+006,
#                5.330197e+006, 5.348247e+006,
#                5.369642e+006]
kimberlines = []
mod.origin = data.origin
# data.locations = np.array([[0 for ii in range(17)],
#                            [0.000000000E+00, 0.501143799E+04, 0.104698379E+05,
#                             0.136017852E+05, 0.178389980E+05, 0.208527168E+05,
#                             0.247133633E+05, 0.279987383E+05, 0.328820195E+05,
#                             0.345390352E+05, 0.372428438E+05, 0.394137422E+05,
#                             0.433482109E+05, 0.467561680E+05, 0.507770469E+05,
#                             0.534360625E+05, 0.653211367E+05]]) / 1000
# mod.to_UTM()
# mod.to_latlong('10N')
# data.rotate_sites(azi=-14)
# data.locations = data.get_locs(mode='centered')
# mod.to_latlong('13W')
if mod.coord_system == 'UTM':
    mod.dx = [xx / 1000 for xx in mod.dx]
    mod.dy = [yy / 1000 for yy in mod.dy]
slice_num = 34
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 3
file_path = 'C:/Users/eroots/phd/ownCloud/Documents/Dryden_paper/RoughFigures/'
file_name = 'dryden_plan_gray.png'
title_ = 'Standard Inversion'
save_fig = 0
use_alpha = 1
saturation = 0.8
lightness = 0.4

# xlim = [min([ix for ix in mod.dx if ix <= 5250000], key=lambda x: abs(mod.dx[x] - 5250000)),
#         min([iy for iy in mod.dy if iy >= 5450000], key=lambda x: abs(mod.dx[x] - 5450000))]
# xlim = [5250000, 5450000]
# zlim = [0, 200]
xlim = [-100, 100]
# zlim = [-100, 100]
zlim = [0, 50]
lut = 64
cax = [1, 5]
isolum = False
# xlim = [-123.5, -121.5]
# xlim = [-7, 74]
# zlim = [0, 5]
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
cmap_name = 'jet_r'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
# cmap_name = 'gray'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'

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

# vals = np.log10(mod.vals[:, 31, :])
# vals = np.log10(mod.vals[:, :, :])
# vals = np.log10(mod.vals[:, 73, :])
# vals = np.log10(mod.vals[11, :, :])
# vals = np.log10(mod.vals[:, :, 30])
vals = np.log10(mod.vals[:, slice_num, :]).T
#  Important step. Since we are normalizing values to fit into the colour map,
#  we first have to threshold to make sure our colourbar later will make sense.
vals[vals < cax[0]] = cax[0]
vals[vals > cax[1]] = cax[1]
if use_alpha:
    alpha = normalize_resolution(mod, reso)
    # alpha = alpha[11, :, :]
    alpha = alpha[:, slice_num, :]
if mode == 2:
    vals = interpolate_slice(y, z, vals, 300)
    if use_alpha:
        alpha = interpolate_slice(y, z, alpha, 300)

norm_vals = (vals - np.min(vals)) / \
            (np.max(vals) - np.min(vals))
if mode == 2:
    rgb = cmap(np.flipud(norm_vals.T))
    if use_alpha:
        alpha = np.flipud(alpha.T)
else:
    rgb = cmap(norm_vals)
    if use_alpha:
        alpha = alpha.T
# Just turn the bottom row transparent, since those cells often somehow still have resolution
rgba = copy.deepcopy(rgb)
if use_alpha:
    # alpha[-1, :] = alpha[-1, :].clip(max=0.5)
    # alpha[:, :] = 1
    rgba[..., -1] = alpha


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

fig = plt.figure(figsize=(15, 10))
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
    ax = fig.add_subplot(1, 1, 1)
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
        im, ax = pcolorimage(ax, x=(np.array(mod.dx)) / 1000,
                             y=np.array(mod.dz) / 1000,
                             A=(to_plot), cmap=cmap)
        # plt.plot(data.locations[:, 1] / 1000, data.locations[:, 0] / 1000, 'k.')
        plt.plot(data.locations[:, 1] / 1000, np.zeros((data.locations[:, 1].shape)), 'k.')
        # for jj, site in enumerate(data.site_names):
        #     plt.annotate(site,
        #                  xy=(data.locations[jj, 1] / 1000, data.locations[jj, 0] / 1000),
        #                  color='w')
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
ax.tick_params(axis='both', labelsize=18)
# locs = ax.plot(data.locations[1, :], np.zeros((data.locations.shape[1])) - 0.05, 'kv', markersize=10)[0]
# locs.set_clip_on(False)
# for label in ax.xaxis.get_ticklabels():
#     label.set_visible(False)
# for label in ax.yaxis.get_ticklabels():
#     label.set_visible(False)
# ax.tick_params(axis='y', labelsize=10)
fig.subplots_adjust(right=0.8)
cb_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cmap=cmap, cax=cb_ax)
cb.set_clim(cax[0], cax[1])
cb.ax.tick_params(labelsize=12)
cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
             rotation=270,
             labelpad=20,
             fontsize=18)
cb.draw_all()

fig.set_dpi(300)
# ax.set_aspect(3)
if save_fig:
    fig.savefig(file_path + file_name, dpi=1200,
                transparent=True, pad_inches=3)
plt.show()
