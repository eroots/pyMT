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
# import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes
# from scipy.interpolate import SmoothBivariateSpline as RBS
import copy
# import colorcet as cc
import colorsys
from pyMT.e_colours import colourmaps
import sys


#local_path = 'C:/Users/eroots'
#local_path = 'C:/Users/eric/'
local_path = 'E:/'


def check_UTM(UTM):
    UTM_letter = UTM[-1]
    if len(UTM) == 3:
        UTM_number = int(UTM[:2])
    elif len(UTM) == 2:
        UTM_number = int(UTM[0])
    return UTM_number, UTM_letter


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


def project_locations(locations, zone, letter):
    # data.locations = data.get_locs(mode='latlong')
    for ii in range(len(locations)):
        easting, northing = utils.project((locations[ii, 1],
                                           locations[ii, 0]),
                                          zone=zone, letter=letter)[2:]
        locations[ii, 1], locations[ii, 0] = easting, northing
    return locations


# File paths
# main_transect: list file containing sites through which you want to slice
# data: list file or ModEM data file used for inversion. Use a list file if you want coordinates to be in UTM
# backup_data: Copy-paste 'data' here. Just a hack, will fix later.
    # NOTE:
    # If you are using ModEM data files, you must change 'RawData' to 'Data'; some features might not be available in this case (e.g., location projection)
# mod: the model file
# seismic: Alternative to using a list file to define the slice, you could load a csv file with locations here
#          e.g., a seismic line. Assumes you have 3 columns 'trace', 'x', 'y'
# main_transect = WSDS.RawData('filename')
# data = WSDS.RawData('filename')
# backup_data = WSDS.RawData('filename')
# mod = WSDS.Model('filename')
main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb_NS.lst')
data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
backup_data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static.model')
# TTZ
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/ttz_south.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/allsites.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/allsites.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/TTZ/full_run/ZK/1D/ttz1D-all_lastIter.rho')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/TTZ/full_run/ZK/hs2500/ttz_NLCG_072.rho')
# Define a UTM zone to project to. Use None if you want to use the default, or if you are using a ModEM data file
UTM_zone = None
# UTM_zone = '16U'
# seismic = pd.read_table('file_name',
                        # header=0, names=('trace', 'x', 'y'), sep='\s+')
# How to define the slice. 1 is through MT stations, 2 is through points in a csv (previously 'seismic'), and 3 is through points given by 'slice_points_x' and 'slice_points_y'
transect_types = {1: 'mt', 2: 'csv', 3: 'points'}
transect_type = 3  # Set to 1, 2, or 3
slice_points_x = main_transect.locations[:, 1]
slice_points_y = main_transect.locations[:, 0]
slice_points_x = (653376, 653376)
slice_points_y = (5332020, 5375000)
# slice_points_x = (585170, 607599)
# slice_points_y = (5323280, 5330550)
points_in_latlong = 0 # Set to true if specifying slice_points in latlong
xaxis_increasing = 1 # Force x-axis to be increasing (1) or decreasing (0)
use_trace_ticks = 0  # If using seismic, do you want the x-axis to be CDP values?
azi = 0  # Rotation angle for model (not well tested)
reso = []  # Include resolution file?
### Number of interpolation points between each station. 
### This is the number per pair of stations points, so turn this up if you're only using a few points
ninterp = 200
nz_interp = 5
padding = 25000  # Padding (in m) at the ends of the slice
ninterp_padding = 100  # Number of interpolation points in the padding
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}  # Image style. Use 3 if you're not sure.
mode = 3

# Figure save options
file_path = 'E:/phd/NextCloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/alongSeis/line14/'
# file_name = 'ttz-1D-ZK_south_line'
file_name = 'AG_straighSN-L14_60kmDeep-Linear_turbo0-4.5'
file_types = ['.png']#, '.svg']  # File save format(s)
title_ = 'Standard Inversion'  # Title of plot
rotate_back = 0  # If data is rotated, do you want to rotate it back to 0?
linear_xaxis = 1  # Use a linear x-axis or keep in easting-northing? Recommended if using a irregular slice
plot_direction = 'sn'
save_fig = 1  # Save the figure?
save_dat = 0  # Save the plotted slice as a csv?
annotate_sites = 0  # Plot station names? 
site_markers = 0  # Include site markers (from main_transect) on plot? This is turned off if you aren't using transect_type 1
plot_map = 1  # Plot a map with all stations (black) and main_transect (red)?
plot_contours = 0
contour_levels = [1, 2, 3, 4]
dpi = 600  # DPI of saved image
csv_name = 'E:/phd/NextCloud/Documents/ME_Transects/Malartic/RoughFigures/Mal_R1_slices/along_transect_turbo/w_contours/'
# csv_name = local_path + '/phd/Nextcloud/Metal Earth/Data/model_csvs/rouyn_alongMT.dat'
use_alpha = 0  # Apply alpha according to resolution file?
saturation = 0.8  # Colour parameters. Leave as is, or play around
lightness = 0.4
xlim = []  # x-axis limits
zlim = [0, 60]  # y-axis limits
aspect_ratio = 1  # aspect ratio of plot
lut = 64  # number of colour values
cax = [0, 4.5]  # Colour axis limits
isolum = 0  # Apply isoluminate normalization?

# For more complicated slicing
# Input a set of sites you would like to 'nudge' a set distance off the main transect.
# nudge_sites goes east, reverse_nudge goes west.
# If you are using 'points' to define the slice, nudges are applied to all points
# nudge_sites = ['']
nudge_sites = main_transect.site_names
reverse_nudge = ['']
use_nudge = 0 # Do you want to use the nudges?

# Choose color map from below
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
# cmap_name = 'turbo_r'
cmap_name = 'turbo_r_mod'
# cmap_name = 'gray'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'
# cmap_name = 'jetplus'
force_NS = 1  # Force the slice to plot south-to-north
fig_num = 0
if UTM_zone:
    try:
        UTM_number, UTM_letter = check_UTM(UTM_zone)
        main_transect.to_utm(letter=UTM_letter, zone=UTM_number)
        data.to_utm(letter=UTM_letter, zone=UTM_number)
        backup_data.to_utm(letter=UTM_letter, zone=UTM_number)
    except AttributeError:
        print('All data must be RawData (read from EDIs or j-format) to project to a UTM zone. Please try again.')
        sys.exit()
all_backups = {'model': copy.deepcopy(mod), 'main_transect': copy.deepcopy(main_transect),
               'data': copy.deepcopy(data), 'backup_data': copy.deepcopy(backup_data)}
# for nudge_dist in range(-30000, 32500, 2500):  # Modify this as needed. You could use this to do slices through your transect at offsets
for nudge_dist in [0]:
    # file_name = 'mal_along_MT_turbo0-5_{}m-offset'.format(nudge_dist)
    mod = copy.deepcopy(all_backups['model'])
    data = copy.deepcopy(all_backups['data'])
    main_transect = copy.deepcopy(all_backups['main_transect'])
    backup_data = copy.deepcopy(all_backups['backup_data'])
    fig_num += 1
    main_transect.locations = main_transect.locations[main_transect.locations[:, 0].argsort()]
    # Sort the site names so the same is true
    main_transect.site_names = sorted(main_transect.site_names,
                                      key=lambda x: main_transect.sites[x].locations['X'])
    if transect_types[transect_type] == 'points':
        nudge_locations = np.array((slice_points_y, slice_points_x)).T
        if points_in_latlong and UTM_zone:
            nudge_locations = project_locations(nudge_locations, letter=UTM_letter, zone=UTM_number)
        if plot_direction.lower  == 'sn':
            nudge_locations = nudge_locations[nudge_locations[:, 0].argsort()]
        elif plot_direction.lower  == 'we':
            nudge_locations = nudge_locations[nudge_locations[:, 1].argsort()]
        # site_markers = 1
    elif transect_types[transect_type] == 'mt':
        nudge_locations = copy.deepcopy(main_transect.locations)
    if use_nudge:
        if transect_types[transect_type] == 'points':
            nudge_locations[:, 1] += nudge_dist
        else:
            for ii, site in enumerate(main_transect.site_names):
                if (site in nudge_sites):
                    nudge_locations[ii, 1] += nudge_dist
                elif site in reverse_nudge:
                    nudge_locations[ii, 1] -= nudge_dist
    if plot_map and transect_types[transect_type] == 'mt':
        site_x, site_y = [main_transect.locations[:, 1],
                          main_transect.locations[:, 0]]
        qx_map, qy_map = [], []
        X = np.linspace(nudge_locations[0, 0] - padding, nudge_locations[0, 0], ninterp_padding)
        Y = np.interp(X, site_y, site_x)
        qx_map.append(Y)
        qy_map.append(X)
        for ii in range(len(nudge_locations) - 1):
        # for ii, site in enumerate(main_transect.site_names[:-1]):
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
    elif plot_map and transect_types[transect_type] == 'points':
        qy_map, qx_map = nudge_locations.T

    data = copy.deepcopy(main_transect)
    if azi:
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
    if transect_types[transect_type] == 'seismic':
            qx, qy = (np.array(seismic['x'] / 1000),
                      np.array(seismic['y']) / 1000)
            if azi:
                locs = utils.rotate_locs(np.array((qy, qx)).T, azi)
                qx, qy = locs[:, 1], locs[:, 0]
            if plot_map:
                qx_map = copy.deepcopy(qx) * 1000
                qy_map = copy.deepcopy(qy) * 1000
    elif transect_types[transect_type] == 'mt':
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
        qx = np.concatenate(qx).ravel() / 1000
        qy = np.concatenate(qy).ravel() / 1000
    else:
        qx = []
        qy = []
        # for ii in range(len(data.locations[:, 0]) - 1):
        if plot_direction == 'sn':
            use_x, use_y = nudge_locations[:, 1], nudge_locations[:, 0]
        else:
            use_x, use_y = nudge_locations[:, 0], nudge_locations[:, 1]
        for ii in range(len(slice_points_x) - 1):
            X = np.linspace(use_y[ii], use_y[ii + 1], ninterp)
            Y = np.interp(X, use_y[:], use_x[:])
            qx.append(Y)
            qy.append(X)
        qx = np.concatenate(qx).ravel() / 1000
        qy = np.concatenate(qy).ravel() / 1000
    x, y, z = [np.zeros((len(mod.dx) - 1)),
               np.zeros((len(mod.dy) - 1)),
               np.zeros((len(mod.dz) - 1))]
    for ii in range(len(mod.dx) - 1):
        x[ii] = (mod.dx[ii] + mod.dx[ii + 1]) / 2
    for ii in range(len(mod.dy) - 1):
        y[ii] = (mod.dy[ii] + mod.dy[ii + 1]) / 2
    for ii in range(len(mod.dz) - 1):
        z[ii] = (mod.dz[ii] + mod.dz[ii + 1]) / 2
    if cmap_name in ('jetplus', 'turbo', 'turbo_r', 'turbo_r_mod'):
        cmap = colourmaps.get_cmap(cmap_name, lut)
    else:
        cmap = cm.get_cmap(cmap_name, lut)
    qz = []
    for ii in range(len(z) - 1):
        qz.append(np.linspace(z[ii], z[ii + 1], nz_interp))
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
    if plot_direction == 'we':
            qx, qy = qy, qx
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
    if reso and use_alpha:
        alpha[-1, :] = alpha[-1, :].clip(max=0.5)
        rgba[..., -1] = alpha

    # Rotate locations back to true coordinates
    if azi and rotate_back:
        data.locations = utils.rotate_locs(data.locations, azi=-azi)
        p = utils.rotate_locs(np.array((qx, qy)).T, azi=azi)
        qx_rot = p[:, 0]
        qy_rot = p[:, 1]
    else:
        qy_rot = qy
        qx_rot = qx
    if linear_xaxis:
        linear_x = np.zeros(qx_rot.shape)
        linear_x[1:] = np.sqrt((qx_rot[1:] - qx_rot[:-1]) ** 2 + (qy_rot[1:] - qy_rot[:-1]) ** 2)
        linear_x = np.cumsum(linear_x)
        nodes = np.array([qy_rot * 1000, qx_rot * 1000]).T
        if transect_types[transect_type] == 'mt':
            linear_site = np.zeros((len(data.locations)))
            use_locs = data.locations
        else:
            linear_site = np.zeros((len(nudge_locations)))
            use_locs = nudge_locations
        for ii, (x, y) in enumerate(use_locs):
            if use_nudge and transect_types[transect_type] == 'points':
                y += nudge_dist
            elif use_nudge and data.site_names[ii] in nudge_sites:
                y += nudge_dist
            dist = np.sum((nodes - np.array([x, y])) ** 2, axis=1)
            idx = np.argmin(dist)
            linear_site[ii] = linear_x[idx]
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

        cmap = colors.ListedColormap(cmap, name='test1')

    fig = plt.figure(fig_num, figsize=(12, 8))
    ax = fig.add_subplot(111)
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
            if linear_xaxis:
                x_axis = linear_x
            else:
                if plot_direction == 'sn':
                    x_axis = qy_rot
                else:
                    x_axis = qx_rot
            to_plot = to_plot[1:, 1:]
            if not xaxis_increasing:
                x_axis = np.flipud(x_axis)
                to_plot = np.flip(to_plot, 1)
            if transect_types[transect_type] == 'seismic' and use_trace_ticks:
                plt.xticks(seismic['trace'][::500])
                im, ax = pcolorimage(ax,
                                     x=(np.array(seismic['trace'])),
                                     y=np.array(qz),
                                     A=(to_plot), cmap=cmap)
            else:
                im, ax = pcolorimage(ax,
                                     x=(np.array(x_axis)),
                                     y=np.array(qz),
                                     A=(to_plot), cmap=cmap)
                ax.set_aspect(aspect_ratio)
            # sites[0].set_clip_on(False)
                if xlim:
                    ax.set_xlim(xlim)
        if zlim:
            ax.set_ylim(zlim)
        ax.invert_yaxis()
        if plot_contours:
            if transect_types[transect_type] == 'seismic':
                X, Y = np.meshgrid(np.array(seismic['trace']), np.array(qz))
                ax.contour(X, Y, vals, levels=contour_levels, colors='k', vmin=cax[0], vmax=cax[1])
            else:
                X, Y = np.meshgrid(x_axis, qz)
                contours = ax.contour(X, Y, vals.T, levels=contour_levels, colors='k', vmin=cax[0], vmax=cax[1])
            ax.clabel(contours, inline=1, fmt='%1.0f')
        fig.canvas.draw()
    if linear_xaxis:
        site_x = linear_site
        ax.set_xlabel('Distance (km)', fontsize=14)
    else:
        if plot_direction == 'sn':
            site_x = data.locations[:, 0] / 1000
        else:
            site_x = data.locations[:, 1] / 1000
    ax.autoscale_view(tight=True)
    ax.tick_params(axis='both', labelsize=14)
    if site_markers:
        locs = ax.plot(site_x,
                       np.zeros((data.locations.shape[0])) - zlim[1] / 100,
                       'kv', markersize=6)[0]
        locs.set_clip_on(False)
    if annotate_sites:
        for jj, site in enumerate(data.site_names):
            if site.startswith('18-'):
                site = site[3:]
            plt.text(s=site,
                     x=site_x[jj],
                     y=-zlim[1] / 5,
                     color='k',
                     rotation=45)
    if plot_map:
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        # fig2.add_axes(ax2)
        ax2.plot(backup_data.locations[:, 1], backup_data.locations[:, 0], 'kv', markersize=6)
        ax2.plot(qx_map, qy_map, 'r--')
        ax2.plot(main_transect.locations[:, 1], main_transect.locations[:, 0], 'rv', markersize=6)
        ax2.set_aspect('equal')
        # ax2.set_xlim([600000, 675000])
        fig2.canvas.draw()
    if save_fig:
        # plt.show()
        for ext in file_types:
            fig.savefig(file_path + file_name + ext, dpi=dpi,
                        transparent=True)
        fig.clear()
        ax.clear()
        # fig.gcf()
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.pause(1)
        # plt.close(2)
    else:
        plt.show()

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

