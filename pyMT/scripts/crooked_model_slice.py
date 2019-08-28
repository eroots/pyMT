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


local_path = 'C:/Users/eric/'


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


########################################################
# WESTERN SUPERIOR
data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_cull1.lst')
backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_cull1.lst')
mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/wstFinish_lastIter.rho')
main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# seismic = pd.read_table('C:/Users/eric/Desktop/andy/WS1-cdp.dat.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
seismic = pd.read_table(local_path + '/phd/ownCloud/andy/navout_600m.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
#########################################################
# SWAYZE
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# # # # # # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/finish/swz_finish.model')
# # # # # main_transect = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst')
# # # # data = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/WS_J/swzFinish_lastIter_smaller.model')
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/swzPT_lastIter.rho')
# # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/swzPT_lastIter.rho')
# reso = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/WS_J/swzFinish_lastIter_smaller_Resolution.model')
# seismic = pd.read_table(r'C:\Users\eroots\phd\ownCloud\Metal Earth\Data\Seismic\Swayze\Plots\Shapefiles\SWAYZ_LN241_R1_KMIG_SUGETHW_UTM.txt',
                        # header=0, names=('trace', 'x', 'y'), sep='\s+')
# mod.vals = np.log10(mod.vals) - np.log10(mod2.vals)
#########################################################
# Northern Swayze
# main_transect = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/main_transect_north.lst')
# data = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R2north_cull3.lst')
# backup_data = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R2north_cull3.lst')
# mod = WSDS.Model('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/swayze/R1North_1/finish/finish2_morePers_lastIter.rho')
#########################################################
# DRYDEN-ATIKOKAN
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# # # backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst') 
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/mesh/dry5norot_mesh_lastIter.rho')
# seismic = pd.read_table(local_path + '/phd/ownCloud/Metal Earth/Data/Seismic/Dryden/Plots/Shapefiles/DRYDEN_LN341_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# # # main_transect = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# # # data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# backup_data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst') 
# # mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/bg800/dry5norot_lastIter.rho')
# reso = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/mesh/finish/dry5Finish_resolution.model')
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
# main_transect = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
# mod = WSDS.Model('C:/Users/eric/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/moresites/finish/mal5_lastIter.rho')
# seismic = pd.read_table(r'C:\Users\eric\phd\ownCloud\Metal Earth\Data\Seismic\ME_Seismic_PostStack_Migrated_Sections\MAL_LN131_R1_KMIG\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/mal3_lastIter.rho')

# mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/Hex2Mod/HexMal_Z.model')
# main_transect = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_hex.lst')
# backup_data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_hex.lst')

 # seismic = pd.read_table(r'C:\Users\eroots\Downloads\Malartic Seismic Receivers location (1)\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic = pd.read_table(r'C:\Users\eroots\Downloads\Malartic Seismic Receivers location (1)\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# A-G / ROUYN
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/rouyn/j2/main_transect_more.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/j2/center_fewer3.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/j2/center_fewer3.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/centerMore_ModEM/NLCG_NLCG_120.rho')
# seismic = pd.read_table('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/rouyn/rou_seismic_traces.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# MATHESON
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_westLine.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_bb_cull1.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/mat_bb_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/matheson/mat3/mat3_lastIter.rho')
main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/MATBB.lst')
data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/matheson/Hex2Mod/HexMat_all.model')
#########################################################
# AFTON
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/l0.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_cull1.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton1/afton2_lastIter.rho')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton_rot/aftonrot_NLCG_069.rho')
#########################################################
# GERALDTON
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/main_transect.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/geraldton/ger_cull4/ger_NLCG_155.rho')
# seismic = pd.read_table(r'C:/Users/eroots/phd/ownCloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' + 
#                         r'GERALDTON_LN301_R1_KMIG/GERALDTON_LN301_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# main_transect = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/main_transect.lst')
# data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# backup_data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/geraldton/ger_cull4/finish/gerFinish_lastIter.rho')
# seismic = pd.read_table(r'C:/Users/eroots/phd/ownCloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
#                         r'GERALDTON_LN301_R1_KMIG/GERALDTON_LN301_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# LARDER
# main_transect = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/main_transect.lst')
# data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/test.lst')
# backup_data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/j2/test.lst')
# mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/larder/Hex2Mod/Hex2Mod_all.model')
# seismic = pd.read_table(local_path + '/phd/ownCloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
#                         r'LARD_LN321_R1_KMIG/LARD_LN321_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
use_seismic = 0
# azi = 35  # Dryden-regional
# azi = -15  # Swayze regional
azi = 0  # Malartic regional
# UTM_number = 16
# UTM_letter = 'U'
UTM_number = 16
UTM_letter = 'U'
# UTM_letter = 'U'
# padding = 25000
reso = []
ninterp = 20
padding = 10000
ninterp_padding = 50
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 3
# file_path = local_path + '/phd/ownCloud/Documents/ME_Transects/Dryden_paper/RoughFigures/'
# file_name = 'dry2_noOOQ_linear_jet0-5_siteMarkers_alongATTSeis'
# file_path = r'C:/Users/eroots/phd/ownCloud/Documents/Swayze_paper/RoughFigures/'
# file_path = r'C:/Users/eroots/phd/ownCloud/Documents/Geraldton/RoughFigures/'
# file_name = 'swz_norotMeshFinish_linear_bgy_siteMarkers'
# file_path = r'C:/Users/eroots/phd/ownCloud/Documents/Malartic/RoughFigures/')
# file_path = r'C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton1/Report/profiles/'
# file_name = 'aftonrot_l0'
# file_name = 'mal_bg800_nudgeWest_jet0-5_siteMarkers'
# file_name = 'swz_norotFinish_nudge5000_linear_jet1-5'
# file_name = 'gercull4_iter155_linear_jet0-5'
# file_path = local_path + 'phd/ownCloud/Documents/ME_Transects/Dryden_paper/RoughFigures/Dry_R1_slices/'
file_path = local_path + 'phd/ownCloud/Documents/ME_Transects/Dryden_paper/RoughFigures/Dry_R1_slices/'
# file_path = local_path + 'phd/ownCloud/Documents/ME_Transects/Geraldton/RoughFigures/Ger_R1_slices/'
# file_name = 'gerCull4_Finish_jet0-5_nudge5kmEast'
file_name = 'dry_norotMeshFinish_linear_jet0-5'
# file_path = local_path + '/phd/ownCloud/Documents/ME_Transects/Swayze_paper/RoughFigures/'
# file_path = local_path + '/phd/ownCloud/Documents/ME_transects/Malartic/RoughFigures/Mal_R1_slices/'
# file_name = 'swz_norot_linear_jet0-5'
# file_path = r'C:/Users/eroots/phd/ownCloud/Documents/Malartic/RoughFigures/')
# file_path = r'C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton1/Report/profiles/')
# file_name = 'Dry_norotMesh_jet0-5_resolution'
# file_name = 'swz_norotFinish_nudge5000_linear_jet1-5'
# file_name = 'MAL_Hex_ChicobiR2_jet0-5_siteAnnotations'
# file_name = 'MAL_Hex_ChicobiR2_jet0-5'
# file_types = ['.pdf', '.png']
file_types = ['.png']
title_ = 'Standard Inversion'
rotate_back = 0
linear_xaxis = True

save_fig = 0
save_dat = 0
annotate_sites = 0
site_markers = 1
plot_map = 0
site_markers = 0
plot_map = 1
dpi = 600
csv_name = 'C:/Users/eroots/phd/ownCloud/Metal Earth/Data/model_csvs/swayze_regional.dat'
use_alpha = 0
saturation = 0.8
lightness = 0.4

xlim = []
zlim = [0, 50]
# zlim = [0, 400]
lut = 32
# zlim = [0, 100]
lut = 64
isolum = False
cax = [0, 5]
isolum = 0
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
cmap_name = 'jet_r'
# cmap_name = 'gray'
# cmap_name = 'viridis_r'
# cmap_name = 'magma_r'
# cmap_name = 'cet_isolum_r'
# cmap_name = 'cet_bgy_r'
# cmap_name = 'jetplus'
# cmap_name = 'Blues'
# cmap_name = 'nipy_spectral_r'
# cmap_name = 'jetplus'
nudge_sites = ['18-dry043m.dat', '18-dry041m', '18-dry040m',
               '18-dry038m.dat', '18-dry037m', '18-dry001m']
# nudge_sites = ['18-swz024m', 'SWZ016M', 'SWZ034M', '18-swz036l',
#                '18-swz006m', '18-swz001m', '18-swz125m']
# nudge_sites = ['MAL007M', 'MAL005M', 'MAL004M']
# nudge_sites = ['GER007M', 'GER006M', 'GER005M', 'GER004M']
reverse_nudge = ['MAL008M', 'MAL009M', 'MAL010L']
nudge_sites = ['18-swz024m', 'SWZ016M', 'SWZ034M', '18-swz036l',
               '18-swz006m', '18-swz001m', '18-swz125m']
# nudge_sites = ['MAL007M', 'MAL005M', 'MAL004M']
# nudge_sites = ['GER009M', 'GER008L', 'GER007M', 'GER006M', 'GER005M', 'GER004M']
# nudge_sites = main_transect.site_names
# reverse_nudge = ['MAL008M', 'MAL009M', 'MAL010L']
reverse_nudge = []

nudge_dist = 5000
use_nudge = 1

# data.to_utm(UTM_number, UTM_letter)
# main_transect.to_utm(UTM_number, UTM_letter)
# backup_data.to_utm(UTM_number, UTM_letter)

# Make sure the sites go north-south
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
origin = backup_data.origin
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
    linear_site = np.zeros((len(data.locations)))
    for ii, (x, y) in enumerate(data.locations):
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

fig = plt.figure(1, figsize=(12, 8))
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
        if linear_xaxis:
            x_axis = linear_x
        else:
            x_axis = qy_rot
        to_plot = to_plot[1:, 1:]
        im, ax = pcolorimage(ax,
                             x=(np.array(x_axis)),
                             y=np.array(qz),
                             A=(to_plot), cmap=cmap)

        # sites = ax.plot(data.locations[:, 0] / 1000,
        #                 np.zeros(len(data.locations[:, 1])) - 0.5,
        #                 'wv', mec='k', markersize=7)
        ax.set_aspect('equal')
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
if linear_xaxis:
    site_x = linear_site
    ax.set_xlabel('Distance (km)', fontsize=14)
else:
    site_x = data.locations[:, 0] / 1000
ax.autoscale_view(tight=True)
ax.tick_params(axis='both', labelsize=14)
if site_markers:
    locs = ax.plot(site_x,
                   np.zeros((data.locations.shape[0])) - zlim[1] / 100,
                   'kv', markersize=6)[0]
    locs.set_clip_on(False)
# for jj, site in enumerate(data.site_names):
#     plt.text(s=site,
#              x=data.locations[jj, 0] / 1000,
#              y=-7.5,
#              color='k',
#              rotation=90)

# if site_markers:
#     locs = ax.plot(site_x,
#                    np.zeros((data.locations.shape[0])) - 0.5,
#                    'kv', markersize=6)[0]
    # locs.set_clip_on(False)
if annotate_sites:
    for jj, site in enumerate(data.site_names):
        if site.startswith('18-'):
            site = site[3:]
        plt.text(s=site,
                 x=site_x[jj],
                 y=-zlim[1] / 12,
                 color='k',
                 rotation=45)
if plot_map:
    fig2 = plt.figure(2)
    # win2 = Divider(fig2, (0.1, 0.1, 0.8, 0.8), h, v, aspect=False)
    # ax2 = Axes(fig2, win2.get_position())
    # ax2.set_axes_locator(win2.new_locator(nx=1, ny=1))
    ax2 = fig2.add_subplot(111)
    # fig2.add_axes(ax2)
    ax2.plot(backup_data.locations[:, 1], backup_data.locations[:, 0], 'kv', markersize=6)
    ax2.plot(qx_map, qy_map, 'r--')
    # for jj, site in enumerate(backup_data.site_names):
    for jj, site in enumerate(data.site_names):
        plt.text(s=site,
                 x=data.locations[jj, 1],
                 y=data.locations[jj, 0],
                 color='k')
    ax2.plot(main_transect.locations[:, 1], main_transect.locations[:, 0], 'kv', markersize=6)
    ax2.set_aspect('equal')
    fig2.canvas.draw()
# locs.set_clip_on(False)
# for label in ax.xaxis.get_ticklabels():
#     label.set_visible(False)
# for label in ax.yaxis.get_ticklabels():
#     label.set_visible(False)
# ax.tick_params(axis='y', labelsize=10)
# fig.subplots_adjust(right=0.8)

divider = make_axes_locatable(ax)
cb_ax = divider.append_axes('right', size='2.5%', pad=0.1)
cb = plt.colorbar(im, cmap=cmap, cax=cb_ax, orientation='vertical', extend='both')  # , extend='both'   Gives pointy ends
cb.mappable.set_clim(cax[0], cax[1])
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
