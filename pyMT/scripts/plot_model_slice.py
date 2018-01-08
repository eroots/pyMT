import pyMT.data_structures as WSDS
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.image import PcolorImage
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
import copy
import colorcet as cc
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
    ax.images.append(img)
    ax.set_xlim(left=x[0], right=x[-1])
    ax.set_ylim(bottom=y[0], top=y[-1])
    ax.autoscale_view(tight=True)
    return img


def interpolate_slice(x, y, Z, NP):
    mod_interp = RBS(x, y, Z)
    interp_vals = mod_interp(np.linspace(x[0], x[-1], NP),
                             np.linspace(y[0], y[-1], NP))
    return interp_vals


mod = WSDS.Model(r'C:\Users\eric\Documents' +
                 r'\MATLAB\MATLAB\Inversion\Regions' +
                 r'\abi-gren\New\abi0_sens\outSens_model.00')
reso = WSDS.Model(r'C:\Users\eric\Documents' +
                  r'\MATLAB\MATLAB\Inversion\Regions' +
                  r'\abi-gren\New\abi0_sens\Resolution0_inverted.model')
data = WSDS.RawData(r'C:\Users\eric\Documents' +
                    r'\MATLAB\MATLAB\Inversion\Regions' +
                    r'\abi-gren\New\j2\test.lst',
                    datpath=r'C:\Users\eric\Documents' +
                    r'\MATLAB\MATLAB\Inversion\Regions' +
                    r'\abi-gren\New\j2')
kimberlines = [5.341140e+006, 5.348097e+006,
               5.330197e+006, 5.348247e+006,
               5.369642e+006]
mod.origin = data.origin
mod.to_UTM()
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 2
use_alpha = 0
saturation = 0.8
lightness = 0.45

# xlim = [min([ix for ix in mod.dx if ix <= 5250000], key=lambda x: abs(mod.dx[x] - 5250000)),
#         min([iy for iy in mod.dy if iy >= 5450000], key=lambda x: abs(mod.dx[x] - 5450000))]
xlim = [5250000, 5450000]
zlim = [0, 200]
lut = 64
cax = [1, 5]
isolum = False
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
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

vals = np.log10(mod.vals[:, 31, :])
#  Important step. Since we are normalizing values to fit into the colour map,
#  we first have to threshold to make sure our colourbar later will make sense.
vals[vals < cax[0]] = cax[0]
vals[vals > cax[1]] = cax[1]
alpha = normalize_resolution(mod, reso)
alpha = alpha[:, 31, :]
if mode == 2:
    vals = interpolate_slice(x, z, vals, 500)
    alpha = interpolate_slice(x, z, alpha, 500)

norm_vals = (vals - np.min(vals)) / \
            (np.max(vals) - np.min(vals))
if mode == 2:
    rgb = cmap(np.flipud(norm_vals.T))
    alpha = np.flipud(alpha.T)
else:
    rgb = cmap(norm_vals.T)
    alpha = alpha.T
rgba = copy.deepcopy(rgb)
rgba[..., -1] = alpha
# cmap[..., -1] = reso.vals[:, 31, :]

if isolum:
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
    hls[:, :, 1] = alpha.clip(min=0.2, max=lightness)
    rgba_l = hls2rgb(hls)
    hls[:, :, 1] = np.abs(1 - alpha).clip(min=lightness, max=0.8)
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

fig = plt.figure(figsize=(8, 8))
# fig = plt.gcf()
for ii in range(1):
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
        im = plt.pcolor(mod.dx, mod.dz, np.log10(mod.vals[:, 31, :]).T,
                        vmin=1, vmax=5, cmap=cmap)
    elif mode == 2:
        im = plt.imshow(to_plot, cmap=cmap, vmin=cax[0], vmax=cax[1], aspect='auto',
                        extent=[x[0], x[-1], z[0] / 1000, z[-1] / 1000])
    elif mode == 3:
        im = pcolorimage(ax, x=np.array(mod.dx) / 1000,
                         y=np.array(mod.dz) / 1000,
                         A=to_plot, cmap=cmap)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('Northing (m)', fontsize=14)
    ax.set_ylabel('Depth (km)', fontsize=14)
    # ax.set_title(title)
    for line in kimberlines:
        ax.plot([line, line], [0, 200], 'w-', lw=0.5)

ax.autoscale_view(tight=True)
ax.tick_params(axis='both', labelsize=10)
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
# ax.tick_params(axis='y', labelsize=10)
fig.subplots_adjust(right=0.8)
cb_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cmap=cmap, cax=cb_ax)
cb.set_clim(cax[0], cax[1])
cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
             rotation=270,
             labelpad=20,
             fontsize=12)
cb.draw_all()
plt.show()
fig.savefig('profile1_' + cmap_name + '.pdf', dpi=600)

