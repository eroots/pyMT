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
import segyio
import pyproj


#local_path = 'C:/Users/eroots'
#local_path = 'C:/Users/eric/'
local_path = 'E:'
# local_path = 'E:/'


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
# SWAYZE NORTH
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/north_main_transect_all.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R2North_all.lst')
# backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/R2North_all.lst')
# # # # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/R2North_1/hs_Z/finish/northFinish_lastIter.rho')
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/R2North_1/placed_TFPT/northTFPT_lastIter.rho')
##########################################################
# SWAYZE SOUTH
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/south_main_transect_all.lst')
# # data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/south_all_j2.lst')
# # backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/j2/south_all_j2.lst')
# # # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/R2South_new1/hs_Z/finish/southFinish_lastIter.rho')
# # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/swayze/R2South_new1/placed_TFPT/southTFPT_hsRef-2_lastIter.rho')
# ##########################################################
# # SWAYZE
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/main_transect_more.lst')
# data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# backup_data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/swz_cull1.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swz_NConduit-replaced.model')
# seismic = pd.read_table(r'E:\phd\Nextcloud\Metal Earth\Data\Seismic\Swayze\Plots\Shapefiles\SWAYZ_LN241_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# mod.vals = np.log10(mod.vals) - np.log10(mod2.vals)
#########################################################
# DRYDEN-ATIKOKAN
# seismic = pd.read_table(r'C:\Users\eroots\phd\ownCloud\Metal Earth\Data\Seismic\Atikoken\Plots\Shapefiles\ATIKOKAN_LN351_R1_KMIG_SUGETHW_UTM.txt',
                                                # header=0, names=('trace', 'x', 'y'), sep='\s+')
# # main_transect = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# # data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst')
# backup_data = WSDS.RawData(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst') 
# # mod = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/bg800/dry5norot_lastIter.rho')
# reso = WSDS.Model(local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/dry5/norot/mesh/finish/dry5Finish_resolution.model')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/wst2dry-wOOQ_lastIter.rho')
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/wst2dry_wOOQ_cull.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/wst2dry_wOOQ_cull.lst') 
mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/capped10000/from-cullZK/wst2dry-capped_lastIter.rho')
main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/main_transect.lst')
data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst')
backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst')
#########################################################
# DRYDEN - R2
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_central_all.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_central_all.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_central_all.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/dryden/wst2dry2/R2Central/fromDryOnly/R2-fromDryOnly_lastIter.rho')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/Dryden/' + 
#                         r'DRYDEN_LN341_R2_PSTM/Shapefiles/DRYDEN_LN341_R2_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# WESTERN SUPERIOR
seismic = pd.read_table(local_path + '/phd/NextCloud/andy/wsup_cdp-bin-1merge_interp-gaps.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
seismic['x'] -= 100000
# # seismic = pd.read_table(local_path + '/phd/NextCloud/andy/wsup_cdp-bin-2b.dat', header=None, names=('cdp', 'x', 'y', 'z', 'rho'), sep='\s+')
# # seismic['x'] = seismic['x'] - 10000
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst')
# main_transect.locations = main_transect.get_locs(mode='lambert')
# data = copy.deepcopy(main_transect)
# backup_data = copy.deepcopy(main_transect)
# mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.rho')
# mod.origin = main_transect.origin
# mod.to_lambert()
# transformer = pyproj.Transformer.from_crs('epsg:32615', 'epsg:3979')
# # out_x, out_y = np.zeros(len(x)), np.zeros(len(y))
# for ii, (xx, yy) in enumerate(zip(seismic['x'], seismic['y'])):
#     seismic['x'][ii], seismic['y'][ii] = transformer.transform(xx, yy)
# # mod = WSDS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/anisotropic/wstZK_ani_lastIter.zani')
# # main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_cull1.lst')
# # data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_cull1.lst')
# # backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/j2/ME_wst_cull1.lst')
# # # # # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/wst_NLCG_061.rho')
# # # # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/wst_bg1000_lastIter.rho')
# # mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho')
##########################################################
# MALARTIC
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
# backup_data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
# mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/mal1/moresites/finish/mal5_lastIter.rho')
# mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/mal1/bg800/moresites/mal_lastIter.rho')
# seismic = pd.read_table(r'E:\phd\NextCloud\Metal Earth\Data\Seismic\ME_Seismic_PostStack_Migrated_Sections\MAL_LN131_R1_KMIG\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_bb_cull1.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/malartic/mal1/mal3_lastIter.rho')

# mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/malartic/Hex2Mod/HexMal_Z.model')
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/main_transect_more.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_hex.lst')
# backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_hex.lst')

# seismic = pd.read_table(r'C:\Users\eroots\Downloads\Malartic Seismic Receivers location (1)\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic = pd.read_table(r'C:\Users\eroots\Downloads\Malartic Seismic Receivers location (1)\MAL_LN131_R1_KMIG_SUGETHW_UTM.txt', header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# A-G / ROUYN
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/rouyn/j2/main_transect_more.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/j2/center_fewer3.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/j2/center_fewer3.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/abi-gren/centerMore_ModEM/NLCG_NLCG_120.rho')
# seismic = pd.read_table('E:/phd/NextCloud/data/Regions/MetalEarth/rouyn/rou_seismic_traces.txt',
                                                # header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# # MATHESON
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/matheson/j2/mat_eastLine_all.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
# # # # # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/matheson/Hex2Mod/HexMat_all.model')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/matheson/Hex2Mod/HexMat_Z.model')
# seismic = pd.read_table(local_path + '/phd/ownCloud/Metal Earth/Data/Seismic/Matheson/CDP_coordinate.txt',
#                         header=0, names=('trace', 'x', 'y', 'z', 'Datum', 'Rpeg', 'Fold'), sep='\s+')
# seismic = pd.read_table(local_path + '/phd/ownCloud/Metal Earth/Data/Seismic/Matheson/MATHESON_LN261_R2_KMIG_SUGETHW_UTM.txt',
                                                # header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic = pd.read_table(local_path + '/phd/ownCloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/MATHESON_LN261_R1_KMIG/MATHESON_LN261_R1_KMIG_SUGETHW_UTM.txt',
                                                # header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# AFTON
# main_transect = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/l0.lst')
# data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_aroundOre.lst')
# backup_data = WSDS.RawData('C:/Users/eroots/phd/ownCloud/data/Regions/afton/j2/afton_aroundOre.lst')
# mod = WSDS.Model('C:/Users/eroots/phd/ownCloud/data/Regions/afton/afton3/afton3_lastIter.rho')
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
# main_transect = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# backup_data = WSDS.RawData(local_path + '/phd/ownCloud/data/Regions/MetalEarth/geraldton/j2/ger_cull4.lst')
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/geraldton/ger_cull4/finish/gerFinish_lastIter.rho')
# mod = WSDS.Model(local_path + '/phd/ownCloud/data/Regions/MetalEarth/geraldton/ger_cull4/rot25/bg2500/ger_almostLastIter.rho')

# mod = WSDS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/ger_newSites/for_ade/hs3000/s2/gerS2-all_lastIter.rho')
# data = WSDS.Data('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/ger_newSites/for_ade/hs1000/s2/gerS2-all_lastIter.dat')
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/geraldton/j2/main_transect.lst')
# data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/geraldton/j2/ger_2020R1-2.lst')
# backup_data = copy.deepcopy(data)
# # seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
# #                         r'GERALDTON_LN301_R1_KMIG/GERALDTON_LN301_R1_KMIG_SUGETHW_UTM.txt',
# #                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
#                         r'GERALDTON_LN311_R1_KMIG/GERALDTON_LN311_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
# seisline = ['']
#########################################################
# LARDER
# main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb_NS.lst')
# # # main_transect = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/main_transect_amt.lst')
# data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/test.lst')
# backup_data = WSDS.RawData(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/test.lst')
# mod = WSDS.Model(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/Hex2Mod_Z_static.model')
# # # seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/slice_locs.csv', header=0, names=('index', 'trace', 'x', 'y'), sep=',')
# seismic = pd.read_table(local_path + '/phd/NextCloud/Documents/ME_Transects/Larder/LARDERLAKE_LN321_R1_KMIG-20200716T020940Z-001/LARDERLAKE_LN321_R1_KMIG/' +
#                         'Larder_select_slice.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s+')
#########################################################
# UPPER-ABITIBI
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/rouyn/rapolai_2d/Model4_F.csv',
#                         header=0, names=('x', 'y', 'z', 'rho'), sep=',')
# seismic = seismic.loc[np.unique(seismic['x'], return_index=True)[1]]
# seismic['x'] *= 1000
# seismic['y'] *= 1000
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/MALBB_seisNS.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static.model')

# # seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/rouyn/rapolai_2d/Model4_F_corrected.csv',
#                                          # header=0, names=('x','y','z','rho'), sep=',')
# # seismic_all = copy.deepcopy(seismic)
# # seismic = seismic.loc[np.unique(seismic['x'], return_index=True)[1]]
# # seismic = seismic[3:-3]
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/ROUBB_NS.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/Hex2Mod/HexAG_Z_static.model')
# seisline = ['14']
#########################################################
# TTZ
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/ttz_north.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/allsites.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/TTZ/j2/allsites.lst')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/TTZ/full_run/ZK/1D/ttz1D-all_lastIter.rho')
# seisline = ['']
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG-test300ohm_block.model')
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/12.cdp',
#                         header=0, names=('trace', 'x', 'y'), sep='/s+')
# seismic_data_path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/segy/feat_line12_segy/line12_curvelet.sgy'
# seismic = pd.read_table(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/seismic/cdp_utm.dat',
#                         header=0, names=('trace', 'x', 'y', 'z', 'dummy'), sep='/s+')
# seismic_data_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/seismic/merge_lmig_curvelet.sgy'
# seisline = ['12','14','15','16','16a','23','25','28']
# seisline = ['12', '23', '25']
# seisline = ['14', '12', '23', '16a', 'lar']
# seisline = ['12']
# seisline = ['malartic']
# seisline = ['18', '24', '17']
#########################################################
# Golden Triangle
# datpath = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/'
# # main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/line2a.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/sorted_cull1b.lst')
# backup_data = copy.deepcopy(data)
# # backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/all_sorted.lst')
# # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/snorcle/cull1/reErred/norot/sno_lastIter.rho')
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/snorcle/cull1/reErred/wTopo/cull1b/7p5k/sno-wTopo_lastIter.rho')
# # main_transect.remove_sites(sites=['sno_309', 'mt3_3312', 'mt3_3308', 'mt3_3310', 'sno_313'])
# seisline = ['5']
# mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG_Z_only.model')
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/ROUBB.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst')
# # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG_Z_only.model')
# # mod = WSDS.Model(local_path + '/phd/Nextcloud/data/Regions/MetalEarth/Hex2Mod/HexAG-test300ohm_block.model')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
#                         r'ROUYN_LN141_R1_KMIG/ROUYN_LN141_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='/s+')
# seismic_lines = ['17','18','21','24','27']
# seismic_lines = ['14','15','16','16a','25','28']
# seisline = '27'
# seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(seisline),
#                         # header=0, names=('trace', 'x', 'y'),  sep='\s+')
#                         header=0, names=('ffid', 'trace', 'x', 'y', 'z'), sep='\s+')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/Malartic/Plots/Shapefiles/' +
#                         'MAL_LN131_R1_KMIG_SUGETHW_UTM.txt',
#                         header=0, names=('trace', 'x', 'y'), sep='\s')
# seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
                        # r'LARD_LN321_R1_KMIG/LARD_LN321_R1_KMIG_SUGETHW_UTM.txt',
                        # header=0, names=('trace', 'x', 'y'), sep='\s')
####################################################
# PLC (Patterson Lake)
# main_transect = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/plc18/j2/new/line1.lst')
# data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/plc18/j2/new/all.lst')
# backup_data = WSDS.RawData(local_path + '/phd/Nextcloud/data/Regions/plc18/j2/new/all.lst')
####################################################
use_seismic = 1
overlay_seismic = 0
only_seismic = 0
seismic_is_depth = 0
normalize_seismic = 0
clip_val = 0.15
depth_conversion_velocity = 6.3
use_trace_ticks = 1
force_NS = 1
azi = 0  # Malartic regional
# UTM_number = 16
# UTM_letter = 'U'
project_data = True
UTM_number = 15
UTM_letter = 'U'
reso = []
ninterp = 50
nz_interp = 2
interp_method = 'linear'
padding = 10000
ninterp_padding = 10
modes = {1: 'pcolor', 2: 'imshow', 3: 'pcolorimage'}
mode = 3
file_types = ['.png']#, '.svg']
title_ = 'Standard Inversion'
rotate_back = 0
# plot_direction = 'ns'
linear_xaxis = 0
save_fig = 0
save_dat = 0
annotate_sites = 0
site_markers = 0
plot_contours = 0
plot_map = 1
dpi = 300
csv_name = local_path + '/phd/NextCloud/Documents/ME_Transects/wst/slices/wstZK/dats/wstZK-lambert-merged_interpGaps_'
use_alpha = 0
saturation = 0.8
lightness = 0.4
xlim = []
zlim = [0, 100]
aspect_ratio = 1
lut = 32
isolum = False
cax = [0, 5]
isolum = 0
# cmap_name = 'gist_rainbow'
# cmap_name = 'cet_rainbow_r'
# cmap_name = 'jet_r'
# cmap_name = 'bwr_r'
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
# nudge_sites = ['18-dry043m.dat', '18-dry041m', '18-dry040m',
#                '18-dry038m.dat', '18-dry037m', '18-dry001m']
# nudge_sites = ['18-swz024m', 'SWZ016M', 'SWZ034M', '18-swz036l',
                             # '18-swz006m', '18-swz001m', '18-swz125m']
# nudge_sites = ['SWZ016M', '18-swz015a', 'SWZ014M', '18-swz013a',
#                '18-swz012m', '18-swz011a', '18-swz009a']
nudge_sites = ['']
reverse_nudge = ['']
# main_transect.to_utm(zone=17, letter='U')
# nudge_sites = main_transect.sites
# main_transect.remove_sites(sites=['LAR001M', 'LAR002L', 'LAR003M', 'LAR004M', 'LAR036M', 'LAR037L', 'g90009', 'e92005'])
# nudge_sites = ['MAL003M', 'MAL004M', 'MAL002L']
# nudge_sites = ['GER007M', 'GER006M', 'GER005M', 'GER004M']
# reverse_nudge = ['MAL006L', 'MAL007M']
nudge_sites = ['18-swz024m', 'SWZ016M', 'SWZ034M', '18-swz036l',
                             '18-swz006m', '18-swz001m', '18-swz125m']
# nudge_sites = ['MAL007M', 'MAL005M', 'MAL004M']
# nudge_sites = ['GER009M', 'GER008L', 'GER007M', 'GER006M', 'GER005M', 'GER004M']
# nudge_sites = main_transect.site_names
# reverse_nudge = ['MAL008M', 'MAL009M', 'MAL010L']
# nudge_sites = ['SWZ053A', 'SWZ051A', 'SWZ050M', 'SWZ049A',
#                'SWZ048M', 'SWZ047A', 'SWZ046M', 'SWZ045A',
#                'SWZ044M', 'SWZ043A']
# reverse_nudge = ['SWZ056A', 'SWZ067A', 'SWZ068M',
#                  'SWZ069A', 'SWZ070M', 'SWZ071A', 'SWZ065A',
#                  'SWZ072M', 'SWZ073A', 'SWZ074M', 'SWZ075A',
#                  'SWZ076M', 'SWZ066M']

# nudge_dist = 500
# nudge_dist = -750
use_nudge = 1
fig_num = 0
all_backups = {'model': copy.deepcopy(mod), 'main_transect': copy.deepcopy(main_transect),
               'data': copy.deepcopy(data), 'backup_data': copy.deepcopy(backup_data)}
# all_backups = {'main_transect': copy.deepcopy(main_transect),
#                'data': copy.deepcopy(data), 'backup_data': copy.deepcopy(backup_data)}
# for nudge_dist in [-1200, -800, -400, 0, 400, 800, 1200]:
# for nudge_dist in [-1200, -900, -600, -300, 300, 600, 900, 1200]:
# for seisline in seismic_lines:
# for nudge_dist in range(-10000, 12000, 2000):
# rho = [10, 50, 100, 300, 500]
rho = [500]
depth = ['5.0']
# depth = ['5.0', '10.0', '14.0', '23.0']
# for nudge_dist in [5000]:
    # for line in seisline:
# for r in rho:
    # for d in depth:
        # path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/'
        # mod = WSDS.Model(path + '{}ohm/swz_lccTest_{}ohm_{}kmDepth.model'.format(r, r, d))
# seisline = ['line2a.lst', 'line3.lst', 'line5_2b.lst']
# fig_save_path = 'E:/phd/NextCloud/Documents/GoldenTriangle/RoughFigures/model_slices/7p5k_wTopo/'
# fig_save_name = 'line2a'
# fig_save_path = 'E:/phd/NextCloud/Documents/ME_Transects/Geraldton/RoughFigures/model_slices/'
# fig_save_name = 'gerHS3000_alongSeisSouth-linear'
# plot_directions = ['']
plot_direction = 'sn'
seisline = ['dummy'] * 4
fig_save_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/slices/wstZK/100km/cdps/'
# path = 'E:/phd/Nextcloud/data/Regions/plc18/final/feature_test/C1/'
# models = ['C1-500ohm.model', 'C1-500ohm-deep.model', 'C1-1000ohm.model', 'C1-1000ohm-deep.model']
nudge_distances = [0, 0, 0, 0]
# nudge_distances = [6000]
for r in rho:
        # for d in depth:
        for il, line in enumerate(seisline[:1]):
            # all_backups.update({'model': WSDS.Model(path+models[il])})
            # fig_save_name = models[il].replace('.model', '')
            # fig_save_name = line[:-4]
            # path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/'
            # mod = WSDS.Model(path + '{}ohm/swz_lccTest_{}ohm_{}kmDepth.model'.format(r, r, d))
            # for nudge_dist in [nudge_distances[il]]:
            for nudge_dist in [0]:
            # for nudge_dist in range(-50000, 60000, 10000):
                    # line = seisline[0]
                    # for nudge_dist in [0]:
                    # fig_save_path = local_path + '/phd/NextCloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/alongSeis/alternate_cmaps/bwr/'
                    #############
                    # try:
                    #     # int(line)
                    #     if line in ['12', '14']:
                    #             seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(line),
                    #                                                     header=0, names=('trace', 'x', 'y'), sep='\s+')
                    #             seismic['x'][:] = seismic['x'][0]
                    #             seismic['y'] = np.linspace(seismic['y'][0], seismic['y'].iloc[-1], ninterp)
                    #             # seismic_data_path = 'C:/Users/user/Downloads/z_curvelet_recons_pctg_01_Lithoprobe_AG_14_KMIG_satck.sgy'
                    #     else:
                    #         try:
                    #             seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(line),
                    #                                             header=0, names=('ffid', 'trace', 'x', 'y', 'z'), sep='\s+')
                    #             # seismic_data_path = local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/segy/feat_line{}_segy/line{}_curvelet.sgy'.format(line, line)
                                
                        
                    #         except FileNotFoundError:
                    #             seismic = pd.read_table(local_path + '/phd/Nextcloud/Metal Earth/Data/Seismic/ME_Seismic_PostStack_Migrated_sections/' +
                    #                                     r'LARD_LN321_R1_KMIG/LARD_LN321_R1_KMIG_SUGETHW_UTM.txt',
                    #                                     header=0, names=('trace', 'x', 'y'), sep='\s')
                    #         except:
                    #             seismic = pd.read_table(local_path + '/phd/NextCloud/data/Regions/MetalEarth/AG/seismic/rewesternsuperiorandabitibimt/{}.cdp'.format(line),
                    #                                             header=0, names=('trace', 'x', 'y'),  sep='\s+')
                    # except ValueError:
                    #     pass
                    ################
                    mod = copy.deepcopy(all_backups['model'])
                    seismic_data_path = 'D:/OpenDTect_data/RawData/output/line{}_depth.sgy'.format(line)
                    data = copy.deepcopy(all_backups['data'])
                    main_transect = copy.deepcopy(all_backups['main_transect'])
                    # main_transect = WSDS.RawData(datpath + line)
                    backup_data = copy.deepcopy(all_backups['backup_data'])
                    fig_num += 1
                    # fig_save_name = 'AG_alongLitho_linear_line{}_{}_mOffset'.format(line, nudge_dist)
                    # fig_save_name = 'AG_LithoprobeLine{}_depth'.format(line)
                    fig_save_name = 'wstZK-lambert_merged-interpGaps_turbomod0-5_{}m-offset'.format(nudge_dist)
                    if project_data:
                        data.to_utm(UTM_number, UTM_letter)
                        main_transect.to_utm(UTM_number, UTM_letter)
                        backup_data.to_utm(UTM_number, UTM_letter)

                    # # Make sure the sites go north-south
                    # if force_NS:
                    # main_transect.remove_sites(sites=[site for site in main_transect.site_names if 'att' in site.lower()])
                    # main_transect.locations = main_transect.locations[main_transect.locations[:, 0].argsort()]
                    # Sort the site names so the same is true
                    # main_transect.site_names = sorted(main_transect.site_names,
                                                      # key=lambda x: main_transect.sites[x].locations['X'])
                    nudge_locations = copy.deepcopy(main_transect.locations)
                    if use_nudge:
                        for ii, site in enumerate(main_transect.site_names):
                            if site in nudge_sites:
                                    nudge_locations[ii, 1] += nudge_dist
                            elif site in reverse_nudge:
                                    if site == 'MAL007M':
                                            nudge_locations[ii, 1] -= 3000
                                    else:
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
                        if plot_direction == 'sn':
                            data.locations = data.locations[data.locations[:, 0].argsort()]
                        elif plot_direction == 'we':
                            data.locations = data.locations[data.locations[:, 1].argsort()]
                    # A little kludge to make sure the last few sites are in the right order (west-east)
                    # data.locations[1:8, :] = data.locations[np.flip(data.locations[1:8, 1].argsort())]
                    # nudge_locations = copy.deepcopy(data.locations)
                    # for ii, site in enumerate(data.site_names):
                    #     if site in nudge_sites:
                    #         nudge_locations[ii, 1] += nudge_dist
                    if use_seismic:
                        qx, qy = (np.array(seismic['x'] / 1000),
                                            np.array(seismic['y']) / 1000)
                        cdp = np.array(seismic['cdp'])
                        if qy[0] > qy[-1]:
                            qx, qy, cdp = np.flip(qx), np.flip(qy), np.flip(cdp)
                        if azi:
                            locs = utils.rotate_locs(np.array((qy, qx)).T, azi)
                            qx, qy = locs[:, 1], locs[:, 0]
                        if force_NS:
                            if plot_direction == 'sn':
                                if qy[0] > qy[-1]:
                                    qx, qy, cdp = np.flip(qx), np.flip(qy), np.flip(cdp)
                            elif plot_direction == 'we':
                                if qx[0] > qx[-1]:
                                    qx, qy, cdp = np.flip(qx), np.flip(qy), np.flip(cdp)
                            elif plot_direction == 'ns':
                                if qy[0] < qy[-1]:
                                    qx, qy, cdp = np.flip(qx), np.flip(qy), np.flip(cdp)
                            elif plot_direction == 'ew':
                                if qx[0] < qx[-1]:
                                    qx, qy, cdp = np.flip(qx), np.flip(qy), np.flip(cdp)
                        # add = np.arange(1, 30, 0.1)
                        # qx = np.hstack([qx, np.ones(add.shape) * qx[-1]])
                        # qy = np.hstack([qy, add + qy[-1]])
                        # qx = np.hstack([np.ones(add.shape) * qx[0], qx, np.ones(add.shape) * qx[-1]])
                        # qy = np.hstack([qy[0] - np.flip(add), qy, add + qy[-1]])
                        if plot_map:
                            qx_map = copy.deepcopy(qx) * 1000
                            qy_map = copy.deepcopy(qy) * 1000
                        if use_nudge:
                            if plot_direction in ('ns', 'sn'):
                                qx += nudge_dist / 1000
                            else:
                                qy += nudge_dist / 1000
                            if plot_map:
                                if plot_direction in ('ns', 'sn'):
                                    qx_map += nudge_dist / 1000
                                else:
                                    qy += nudge_dist / 1000

                    else:
                        # if plot_direction in ('we', 'ew'):
                        #     nudge_locations = np.fliplr(nudge_locations)
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
                        # if plot_direction in ('we', 'ew'):
                        #     qx, qy = qy, qx
                    kimberlines = []
                    if force_NS:
                        if plot_direction == 'sn':
                            if qy[0] > qy[-1]:
                                qx, qy = np.flip(qx), np.flip(qy)
                        elif plot_direction == 'we':
                            if qx[0] > qx[-1]:
                                qx, qy = np.flip(qx), np.flip(qy)
                        elif plot_direction == 'ns':
                            if qy[0] < qy[-1]:
                                qx, qy = np.flip(qx), np.flip(qy)
                        elif plot_direction == 'ew':
                            if qx[0] < qx[-1]:
                                qx, qy = np.flip(qx), np.flip(qy)
                    x, y, z = [np.zeros((len(mod.dx) - 1)),
                               np.zeros((len(mod.dy) - 1)),
                               np.zeros((len(mod.dz) - 1))]
                    for ii in range(len(mod.dx) - 1):
                            x[ii] = (mod.dx[ii] + mod.dx[ii + 1]) / 2
                    for ii in range(len(mod.dy) - 1):
                            y[ii] = (mod.dy[ii] + mod.dy[ii + 1]) / 2
                    for ii in range(len(mod.dz) - 1):
                            z[ii] = (mod.dz[ii] + mod.dz[ii + 1]) / 2
                    if cmap_name in colourmaps.COLOUR_MAPS:
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
                    if only_seismic:
                            print('Only producing seismic...')
                    else:
                            print('Interpolating...')
                    # vals = griddata(data_points, data_values, query_points, 'nearest')
                            interpolator = RGI((y, x, z), np.transpose(vals, [1, 0, 2]), bounds_error=False, fill_value=5, method=interp_method)
                            # sz = utils.edge2center(seismic_all['z'])
                            # sy = utils.edge2center(seismic_all['y'])
                            # srho = np.array(list(seismic_all['rho']) + [seismic_all['rho'][0]])
                            # interpolator = RGI((sy, sz), 
                            #                     np.reshape(np.array(seismic_all['rho']), [411, 53]),
                                                                    # bounds_error=False, fill_value=5, method=interp_method)
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
                                if use_nudge and data.site_names[ii] in nudge_sites:
                                        y += nudge_dist
                                dist = np.sum((nodes - np.array([x, y])) ** 2, axis=1)
                                idx = np.argmin(dist)
                                linear_site[ii] = linear_x[idx]
                        x_axis = linear_x
                    else:
                        if plot_direction in ('ns', 'sn'):
                                x_axis = qy_rot
                        else:
                                x_axis = qx_rot
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
                    # h = [Size.Fixed(0.), Size.Fixed(6.5)]
                    # v = [Size.Fixed(0.5), Size.Fixed(3.25)]
                    # win = Divider(fig, (0.1, 0.1, 0.8, 0.8), h, v, aspect=False)
                    # ax = Axes(fig, win.get_position())
                    # ax.set_axes_locator(win.new_locator(nx=1, ny=1))
                    # fig.add_axes(ax)
                    # fig = plt.gcf()
                    if not only_seismic:
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
                                        if use_seismic and use_trace_ticks:
                                                # with segyio.open(seismic_data_path, strict=False) as f:
                                                        # cdp = np.array([t[segyio.TraceField.CDP] for t in f.header])
                                                # cdp = range(len(cdp))
                                                aspect_ratio = 'auto'
                                                plt.xticks(cdp[::round(len(cdp)/10)])
                                                im, ax = pcolorimage(ax,
                                                                     x=(np.array(cdp)),
                                                                     y=np.array(qz),
                                                                     A=(to_plot), cmap=cmap)
                                        else:
                                                idx = np.argmin(abs(np.array(qz) - zlim[1]))
                                                im, ax = pcolorimage(ax,
                                                                     x=(np.array(x_axis)),
                                                                     y=np.array(qz),
                                                                     A=(to_plot), cmap=cmap)
                                        # sites = ax.plot(data.locations[:, 0] / 1000,
                                        #                 np.zeros(len(data.locations[:, 1])) - 0.5,
                                        #                 'wv', mec='k', markersize=7)
                                                # ax.set_aspect(aspect_ratio)
                                        # sites[0].set_clip_on(False)
                                                if xlim:
                                                        ax.set_xlim(xlim)
                                if zlim:
                                        ax.set_ylim(zlim)
                                ax.invert_yaxis()
                                if plot_contours:
                                    if use_seismic:
                                        ax.contour(np.arrary((seismic['trace'], qz)), to_plot, levels=contour_levels)
                                    else:
                                        ax.contour(np.arrary((x_axis, qz)), to_plot, levels=contour_levels)
                            # ax.invert_xaxis()
                            # ax.set_xlabel('Latitude', fontsize=20)
                            # ax.set_ylabel('Depth (km)', fontsize=20)
                            # ax.set_title(title_, y=1.02, fontsize=20)
                            
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
                    
                    if overlay_seismic:
                        with segyio.open(seismic_data_path, strict=False) as f:
                                seis_data = np.stack([t.astype(np.float) for t in f.trace])
                                x = np.array([t[segyio.TraceField.CDP_X] for t in f.header])
                                y = np.array([t[segyio.TraceField.CDP_Y] for t in f.header])
                                cdp = np.array([t[segyio.TraceField.CDP] for t in f.header])
                                # header = f.text[0].decode('ascii')
                                dt = f.bin[segyio.BinField.Interval] / 1000
                                samples = f.samples
                        seis_data = seis_data.T
                        if force_NS:
                                if plot_direction == 'sn':
                                        if y[0] > y[-1]:
                                                seis_data = np.fliplr(seis_data)
                                elif plot_direction == 'we':
                                        if x[0] > x[-1]:
                                                seis_data = np.fliplr(seis_data)
                                elif plot_direction == 'ns':
                                        if y[0] < y[-1]:
                                                seis_data = np.fliplr(seis_data)
                                elif plot_direction == 'ew':
                                        if x[0] < x[-1]:
                                                seis_data = np.fliplr(seis_data)
                        # seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
                        # seis_data[:250,:] = 0
                        if normalize_seismic:
                            seis_data = seis_data / np.linalg.norm(seis_data, axis=0)
                        # seis_data = np.fliplr(seis_data)
                        clip = clip_val*np.max(np.abs(seis_data))
                        seis_data[seis_data<-clip] = -clip
                        seis_data[seis_data>clip] = clip
                        seis_data = np.abs(seis_data)
                        alpha = (seis_data) / (np.max(seis_data) * 0.9)
                        if seismic_is_depth:
                                dvec = np.array(samples) / 1000
                        else:
                                tvec = np.arange(seis_data.shape[0]) * dt / 1000
                                dvec = tvec * depth_conversion_velocity / 2
                        ax.imshow((seis_data), 
                                            cmap='gray_r',
                                            extent=[x_axis[0], x_axis[-1], min(dvec[-1], zlim[1]), dvec[0]],
                                            alpha=alpha, interpolation='sinc')
                    ax.set_aspect(aspect_ratio)
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
                    fig.canvas.draw()
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
                                                 y=-zlim[1] / 5,
                                                 color='k',
                                                 rotation=45)
                    if plot_map:
                        fig2 = plt.figure(2)
                        ax2 = fig2.add_subplot(111)
                        # fig2.add_axes(ax2)
                        ax2.plot(backup_data.locations[:, 1], backup_data.locations[:, 0], 'kv', markersize=6)
                        ax2.plot(qx_map, qy_map, 'r--')
                        # for jj, site in enumerate(backup_data.site_names):
                        if annotate_sites:
                            for jj, site in enumerate(data.site_names):
                                plt.text(s=site,
                                                 x=data.locations[jj, 1],
                                                 y=data.locations[jj, 0],
                                                 color='k')
                        ax2.plot(main_transect.locations[:, 1], main_transect.locations[:, 0], 'rv', markersize=6)
                        ax2.set_aspect('equal')
                        # ax2.set_xlim([600000, 675000])
                        fig2.canvas.draw()
                    if save_fig:
                        # plt.show()
                        for ext in file_types:
                                # fig.savefig(fig_save_path + fig_save_name + ext, dpi=dpi,
                                                        # bbox_inches='tight', transparent=True)
                                # fig.tight_layout()
                                fig.savefig(fig_save_path + fig_save_name + ext, dpi=dpi,
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
                # cdp = np.array(range(1, len(qx) + 1))
                cdp = np.tile(cdp[:, np.newaxis], [vals.shape[-1]]).ravel()
                df  = np.array((cdp, x_loc, y_loc, z_loc, np.ravel(vals))).T
                np.savetxt(''.join([csv_name, 'log10.dat']), df, fmt='%12.1f')
                df  = np.array((cdp, x_loc, y_loc, z_loc, 10**np.ravel(vals))).T
                np.savetxt(''.join([csv_name, 'linear.dat']), df, fmt='%12.1f')
                # df = pd.DataFrame(np.array((cdp, x_loc, y_loc, z_loc, np.ravel(vals))).T, columns=None)
                # df[0] = df[0].map(lambda x: '%7.0f' % x)
                # df[1] = df[1].map(lambda x: '%9.1f' % x)
                # df[2] = df[2].map(lambda x: '%9.1f' % x)
                # df[3] = df[3].map(lambda x: '%9.1f' % x)
                # df[4] = df[4].map(lambda x: '%9.2f' % x)
                # df.to_csv(''.join([csv_name, 'log10.dat']), sep=' ', header=None, index=False)
                # df = pd.DataFrame(np.array((cdp, x_loc, y_loc, z_loc, 10 ** (np.ravel(vals)))).T, columns=None)
                # df[0] = df[0].map(lambda x: '%7.0f' % x)
                # df[1] = df[1].map(lambda x: '%9.1f' % x)
                # df[2] = df[2].map(lambda x: '%9.1f' % x)
                # df[3] = df[3].map(lambda x: '%9.1f' % x)
                # df[4] = df[4].map(lambda x: '%9.2f' % x)
                # df.to_csv(''.join([csv_name, 'linear.dat']), sep=' ', header=None, index=False)
                


