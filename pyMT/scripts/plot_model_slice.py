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
from pyMT.e_colours import colourmaps


local_path = 'E:/phd/Nextcloud/'


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
# mod = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.model')
# data = WSDS.Data(r'C:\Users\eroots\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_cull1i_Z.dat')
# reso = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\Resolution.model')
# mod = WSDS.Model(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.model')
# data = WSDS.Data(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_cull1i_Z.dat')
# reso = WSDS.Model(r'C:\Users\eric\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\Resolution.model')
# mod = WSDS.Model(r'C:\Users\eroots\phd\ownCloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\Resolution.model')
# mod = WSDS.Model(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2e_smooth.model')
# data = WSDS.Data(r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2f_bb_Z.dat')
# data = WSDS.Data(datafile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry53.data',
#                  listfile='C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
#####################################################################
# MALARTIC
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/mal3_lastIter.rho')
# data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/mal3_lastIter.dat')
# mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/Hex2Mod/HexMal_Z.model')
# data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_hex.lst')
#####################################################################
# DRYDEN
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/dry5norot_lastIter.rho')
# data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/dry5norot_lastIter.dat')
#####################################################################
# AFTON
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/regions/afton/afton3/afton3_lastIter.rho')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/regions/afton/afton3/afton2_rot25_bg100.rho')
# data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/regions/afton/afton3/afton3_lastIter.dat')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/regions/afton/j2/afton_aroundOre.lst')
##_', str(int(mod.dz[slice_num])))##################################################################
#####################################################################
# LIBEREC
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/regions/Liberec/4site/hs/core/lib_NLCG_006.rho')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/regions/Liberec/4site/hs/core/lib_NLCG_006100.rho')
# data = WSDS.Data('C:/Users/eroots/phd/ownCloud/data/regions/Liberec/4site/hs/core/lib_NLCG_006.dat')
# LARDER
# data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/test.lst')
# # mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/Hex2Mod/Hex2Mod_all.model')
# mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/Hex2Mod/Hex2Mod_Z_static.model')
# reso = []
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/dry53.rho')
# kimberlines = [5.341140e+006, 5.348097e+006,
#                5.330197e+006, 5.348247e+006,
#                5.369642e+006]
######################################################################
# Upper Abitibi
mod = WSDS.Model(local_path + 'data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static.model')
data = WSDS.RawData(local_path + 'data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
site_data = WSDS.RawData(local_path + 'data/Regions/MetalEarth/j2/ROUBB.lst')
kimberlines = []
data.locations = data.get_locs(mode='centered')
locations = np.zeros(site_data.locations.shape)
cc = 0
for ii, site in enumerate(data.site_names):
    if site in site_data.site_names:
        locations[cc, :] = data.locations[ii, :]
        cc += 1
# mod.origin = data.origin
# data.locations = np.array([[0 for ii in range(17)],
#                            [0.000000000E+00, 0.501143799E+04, 0.104698379E+05,
#                             0.136017852E+05, 0.178389980E+05, 0.208527168E+05,
#                             0.247133633E+05, 0.279987383E+05, 0.328820195E+05,
#                             0.345390352E+05, 0.372428438E+05, 0.394137422E+05,
#                             0.433482109E+05, 0.467561680E+05, 0.507770469E+05,
#                             0.534360625E+05, 0.653211367E+05]]) / 1000
# mod.to_UTM()
# mod.to_latlong('10N')
# data.rotate_sites(azi=25)
# data.locations = data.get_locs(mode='centered')
# mod.to_latlong('13W')
# if mod.coord_system == 'UTM':
#     mod.dx = [xx / 1000 for xx in mod.dx]
#     mod.dy = [yy / 1000 for yy in mod.dy]
plane = 'xz'
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 3
# title_ = 'Standard Inversion'
save_fig = 1
use_alpha = 0
saturation = 0.8
lightness = 0.4
annotate_sites = False
site_markers = True
marker = 'kv'
padding = 2000
reverse_xaxis = False
# zlim = [0, 4]
lut = 32
cax = [0, 5]
isolum = False
tick_label_size = 12
axis_label_size = 14
markersize = 5
# slices = [23, 27, 32, 35, 39, 43]  # plan slices afton
# slices = [31, 38, 46, 53, 61]
# lines = ['l0', 'l3', 'l6', 'l9', 'l12']
# slices = [31]
# slices = [37]
slices = list(range(110, 160, 3))
# slices = [110]
lines = ['l0']
xlim = [-60, 50]
zlim = [0, 50]
VE = 1 # Vertical exaggeration
# xlim = [455.5, 778.7]
# zlim = [5.25e6 / 1000, 5.455e6 / 1000]
# xlim, zlim = [], []
# zlim = [0, 5]
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
cmap_name = 'turbo_r'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
# cmap_name = 'gray'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'
# file_path = local_path + 'phd/ownCloud/Documents/ME_Transects/Malartic/RoughFigures/MAL_R1_slices/NS_slices/'
#file_path = local_path + 'phd/ownCloud/Documents/ME_Transects/Larder/RoughFigures/LAR_R1_slices/'
#file_name = ''.join(['LL_ZStatic__jet1-5_NS_', str(int(mod.dy[slice_num])), 'm.png'])
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
    cmap = colourmaps.get_cmap(cmap_name)
else:
    cmap = cm.get_cmap(cmap_name, lut)
file_path = local_path + 'Documents/ME_Transects/Upper_Abitibi/Figures/Rouyn/w_static/NS_slices/'
# for line, slice_num in zip(lines, slices):
for slice_num in slices:
    # title_ = 'Depth: {:<6.2f} m'.format(mod.dz[slice_num])
    title_ = ''
    # file_name = ''.join(['afton3_turbo1-4_', str(int(mod.dz[slice_num])), 'm'])
    file_types = ['.png']
    # locations = data.get_locs(site_list=[site for site in data.site_names if site.startswith(line)])
    # locations = site_data.get_locs()
    # file_name = ''.join(['LL_All_jet1-5_NS_', str(slice_num), 'm.png'])
    file_name = 'ns-slice_{}'.format(int(slice_num))
    # xlim = [min([ix for ix in mod.dx if ix <= 5250000], key=lambda x: abs(mod.dx[x] - 5250000)),
    #         min([iy for iy in mod.dy if iy >= 5450000], key=lambda x: abs(mod.dx[x] - 5450000))]
    # xlim = [5390, 5412]
    # xlim = [5300, 5370]  # Larder Hex
    # xlim = list(np.array([min(data.locations[:, 0]) - 20000, max(data.locations[:, 0]) + 20000]) / 1000)
    # xlim = [5612, 5618]
    # xlim = []
    # zlim = [0, 4]
    # zlim = []
    # xlim = list(np.array([min(data.locations[:, 1]) - padding, max(data.locations[:, 1]) + padding]) / 1000)
    # zlim = list(np.array([min(data.locations[:, 0]) - padding, max(data.locations[:, 0]) + padding]) / 1000)
    # zlim = [0, 200]
    # xlim = [672, 679]
    # zlim = [5610, 5620]

    # vals = np.log10(mod.vals[:, 31, :])
    # vals = np.log10(mod.vals[:, :, :])
    # vals = np.log10(mod.vals[:, 73, :])
    # vals = np.log10(mod.vals[11, :, :])
    # vals = np.log10(mod.vals[:, :, 30])
    if plane.lower() == 'xy':
        vals = np.log10(mod.vals[:, :, slice_num])
        x_ax = np.array(mod.dy)
        y_ax = np.array(mod.dx)
    elif plane.lower() == 'xz':
        vals = np.log10(mod.vals[:, slice_num, :]).T
        x_ax = np.array(mod.dx)
        y_ax = np.array(mod.dz)
    elif plane.lower() == 'yz':
        vals = np.log10(mod.vals[slice_num, :, :]).T
        x_ax = np.array(mod.dy)
        y_ax = np.array(mod.dz)
    #  Important step. Since we are normalizing values to fit into the colour map,
    #  we first have to threshold to make sure our colourbar later will make sense.
    vals[vals < cax[0]] = cax[0]
    vals[vals > cax[1]] = cax[1]
    if use_alpha:
        alpha = normalize_resolution(mod, reso)
        # alpha = alpha[11, :, :]
        alpha = alpha[:, :, slice_num]
    if mode == 2:
        vals = interpolate_slice(y, z, vals, 300)
        if use_alpha:
            alpha = interpolate_slice(y, z, alpha, 300)

    # norm_vals = (vals - np.min(vals)) / \
                # (np.max(vals) - np.min(vals))
    norm_vals = (vals - min(cax[0], np.min(vals))) / \
                (max(np.max(vals), cax[1]) - min(cax[0], np.min(vals)))
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

    fig = plt.figure(1, figsize=(8, 6))
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
            im, ax = pcolorimage(ax, x=x_ax / 1000,
                                 y=y_ax / 1000,
                                 A=(to_plot), cmap=cmap)
            if site_markers:
                #plt.plot(data.locations[:, 1] / 1000, data.locations[:, 0] / 1000, 'kv')
                if plane == 'xz':
                    locs = plt.plot(locations[:, 0] / 1000, np.zeros((locations[:, 0].shape)) - 0.025, marker, markersize=markersize)
                    ax.set_xlabel('Distance (km)', fontsize=axis_label_size)
                    ax.set_ylabel('Depth (km)', fontsize=axis_label_size)
                elif plane == 'yz':
                    locs = plt.plot(locations[:, 1] / 1000, np.zeros((locations[:, 1].shape)) - 0.05, marker, markersize=markersize)
                    ax.set_xlabel('Easting (km)', fontsize=axis_label_size)
                    ax.set_ylabel('Depth (km)', fontsize=axis_label_size)
                elif plane == 'xy':
                    locs = plt.plot(locations[:, 1] / 1000, locations[:, 0] / 1000, marker, markersize=markersize)
                    ax.set_xlabel('Easting (km)', fontsize=axis_label_size)
                    ax.set_ylabel('Northing (km)', fontsize=axis_label_size)
                locs[0].set_clip_on(False)
            if annotate_sites:
                for jj, site in enumerate(data.site_names):
                    plt.text(s=site,
                             x=data.locations[jj, 0] / 1000,
                             y=-zlim[1] / 6,
                             color='k',
                             rotation=90)
                # plt.annotate(site,
                             # xy=(data.locations[jj, 0] / 1000, 0),  #data.locations[jj, 0] / 1000),
                             # color='k')
        if xlim:
            ax.set_xlim(xlim)
        if zlim:
            ax.set_ylim(zlim)
        if plane.lower() in ['xz', 'yz']:
            ax.invert_yaxis()
        if reverse_xaxis:
            ax.invert_xaxis()
        # ax.invert_xaxis()
        # ax.set_xlabel('Easting (km)', fontsize=20)
        # ax.set_xlabel('Distance (km)', fontsize=20)
        # ax.set_xlabel('Northing (km)', fontsize=20)
        # ax.set_ylabel('Depth (km)', fontsize=20)
        ax.set_title(title_, y=1.02, fontsize=20)
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

    # ax.autoscale_view(tight=True)
    ax.tick_params(axis='both', labelsize=tick_label_size)
    # locs = ax.plot(data.locations[:, 0]/1000, np.zeros((data.locations.shape[0])) - 0.2, 'kv', markersize=8)[0]
    # locs.set_clip_on(False)
    # for label in ax.xaxis.get_ticklabels():
    #     label.set_visible(False)
    # for label in ax.yaxis.get_ticklabels():
    #     label.set_visible(False)
    # ax.tick_params(axis='y', labelsize=10)
    # fig.subplots_adjust(right=0.8)
    # cb_ax = fig.add_axes([0.825, 0.15, 0.02, 0.7])
    # cb = fig.colorbar(im, cmap=cmap, cax=cb_ax)
    # cb = fig.colorbar(im, cmap=cmap)
    # cb.set_clim(cax[0], cax[1])
    # cb.ax.tick_params(labelsize=12)
    # cb.set_label(r'$\log_{10}$ Resistivity ($\Omega \cdot m$)',
    #              rotation=270,
    #              labelpad=20,
    #              fontsize=18)
    # cb.draw_all()

    fig.set_dpi(300)
    ax.set_aspect(VE)
    if save_fig:
        for ext in file_types:
           fig.savefig(file_path + file_name + ext, dpi=300,
                       transparent=True)
        plt.close()
    else:
        plt.show()
