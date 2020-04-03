import pyMT.data_structures as WSDS
from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
import pyMT.utils as utils
import matplotlib.pyplot as plt
from pyMT.e_colours import colourmaps as cm
from copy import deepcopy

# local_path = 'C:/Users/eroots/phd/'
local_path = 'E:/phd/Nextcloud'
########################################
# SWAYZE
# model_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.rho'
# base_data = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.dat'
# data_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R2South_4\R2South_4d.data'
# list_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\j2\R2South_4c.lst'
# base_list = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\j2\swz_cull1.lst'
# model_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R1South_4\R1South_4.model'
# data_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R1South_4\R1South_4_placed.data'
#########
# NORTH
# model_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\norot\mesh\PT\swzPT_lastIter.rho'
# base_data = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\norot\mesh\PT\swz_cull1M_TFPT_regErrs.dat'
# data_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2North_1\R2North_all.data'
# list_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\j2\R2North_all.lst'
# base_list = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\j2\swz_cull1.lst'
# model_out = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2North_1\R2North_placed.model'
# data_out = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2North_1\R2North_all_placed.data'

# # SOUTH-EAST
# model_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\norot\mesh\PT\swzPT_lastIter.rho'
# base_data = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\norot\mesh\PT\swz_cull1M_TFPT_regErrs.dat'
# data_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2southeast_1\placed_ZTF\R2Southeast_fix_all.dat'
# list_file = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\j2\R2Southeast_all_NS.lst'
# base_list = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\j2\swz_cull1.lst'
# model_out = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2southeast_1/placed_ZTF\R2Southeast_placed_nest.model'
# data_out = local_path + r'Nextcloud\data\Regions\MetalEarth\swayze\R2southeast_1/placed_ZTF\R2Southeast_all_placed.data'

# model_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\pt\swzPT_lastIter.rho'
# base_data = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\swz_cull1\finish\pt\swz_cull1i_PT.dat'
# data_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R2North_3\R2north_3b_Z.dat'
# list_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\j2\R2North_cull3.lst'
# base_list = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\j2\swz_cull1.lst'
# model_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R2North_3\R2North_3_placed.model'
# data_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\swayze\R2North_3\R2North_3_placed.data'




########################################
# DRYDEN
# data_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/R2central_noOOQ/R2central_noOOQ.dat'
# base_data = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/dry_noOOQ/bg800/finish/dry_noOOQ_wMoreLegacy.dat'
# list_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/R2central_noOOQ.lst'
# base_list = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst'
# model_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/dry_noOOQ/bg800/finish/TFPT/dry2TFPT_lastIter.rho'
# 
# model_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\dryden\R2Central_noOOQ/placed_TFPT/finer/R2Central-2_placed.model'
# data_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\dryden\R2Central_noOOQ/placed_TFPT/finer/R2Central-2_placed.data'
########################################
# WESTERN SUPERIOR - DRYDEN
# model_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# base_data = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.dat'
# base_list = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\wst\j2\ME_wst_usarray.lst'
# data_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/dry5/norot/mesh/finish/dry5_norot.dat'
# # data_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/dry_noOOQ/bg800/finish/dry_noOOQ_wMoreLegacy.dat'
# # list_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst'
# list_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst'
# # # base_list = r'C:\Users\eric\phd\Nextcloud\data\Regions\wst\j2\southcentral.lst'
# model_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\dryden\wst2dry2_wOOQ\wst2dry'
# data_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\dryden\wst2dry2_wOOQ\wst2dry'
########################################
# WESTERN SUPERIOR - STURGEON
model_file = local_path + r'\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
base_data = local_path + r'\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.dat'
base_list = local_path + r'\data\Regions\MetalEarth\wst\j2\ME_wst_usarray.lst'
data_file = local_path + r'\data\Regions\MetalEarth\sturgeon\wst2stu\wst2stu_ffmt_base.dat'
list_file = local_path + r'\data\Regions\MetalEarth\sturgeon\j2\ffmt\stu_use.lst'
model_out = local_path + r'\data\Regions\MetalEarth\sturgeon\wst2stu\stu_all_placed_large.model'
data_out = local_path + r'\data\Regions\MetalEarth\sturgeon\wst2stu\stu_all_placed.data'
########################################
# WESTERN SUPERIOR - GERALDTON
# model_file = local_path + r'Nextcloud\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# base_data = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.dat'
# base_list = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\wst\j2\ME_wst_usarray.lst'
# data_file = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\ger_consCull\fix\R1\ger_consFewer_Z_flagged.dat'
# list_file = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\j2\ger_consCull_R1.lst'
# model_out = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\wst2ger\ger_Z_placed_nest.model'
# data_out = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\wst2ger\ger_Z_placed.data'
########################################
# GERALDTON R2
# model_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/geraldton\wst2ger\wst2ger_lastIter.rho'
# # model_file = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# base_data = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth/geraldton\wst2ger\ger_Z_placed_flagged.dat'
# base_list = r'C:\Users\eroots\phd\Nextcloud\data\Regions\MetalEarth\geraldton\j2\ger_consCull_R1.lst'
# data_file = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\wst2ger\R2\GER_R2_base.data'
# list_file = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\j2\GER_R2.lst'
# model_out = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\wst2ger\R2\ger_R2_placed_nest.model'
# data_out = local_path + r'Nextcloud\data\Regions\MetalEarth\geraldton\wst2ger\R2\ger_R2_placed.data'
# WESTERN SUPERIOR - GERALDTON - R2
# model_file = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/PT/wst2gerPT_lastIter.rho'
# model_file_large = local_path + '/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# # model_file_large = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/PT/ger_Z_placed_large.model'
# base_data = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/PT/wst2gerPT_lastIter.dat'
# base_list = local_path + '/data/Regions/MetalEarth/geraldton/j2/ger_consCull_R1.lst'
# data_file = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/R2/ger_R2_placed_Z_flagged.dat'
# list_file = local_path + '/data/Regions/MetalEarth/geraldton/j2/GER_R2.lst'
# model_out = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/R2/PT/ger_PT_placed_nest.model'
# data_out = local_path + '/data/Regions/MetalEarth/geraldton/wst2ger/R2/PT/ger_Z_placed.data'
########################################
# WESTERN SUPERIOR - DRYDEN - R2
# model_file = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/wst2dry-wOOQPeriods_lastIter.rho'
# model_file_large = local_path + '/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# # model_file_large = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/PT/ger_Z_placed_large.model'
# base_data = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/wst2dry-wOOQPeriods_lastIter.dat'
# base_list = local_path + '/data/Regions/MetalEarth/dryden/j2/wst2dry_wOOQ_cull.lst'
# data_file = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/R2Central/dry_central_base.dat'
# list_file = local_path + '/data/Regions/MetalEarth/dryden/j2/dry_central_all.lst'
# model_out = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/R2Central/dry_all_placed_large.model'
# data_out = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/R2Central/ger_all_placed.data'
########################################
# WESTERN SUPERIOR - DRYDEN - R2
# model_file = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/splitCells/wst2dry-wOOQ_iter49.rho'
# model_file_large = local_path + '/data/Regions/MetalEarth/wst/cull1/bg1000/bigger/ocean/w_usarray/wst_lastIter.rho'
# # model_file_large = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/PT/ger_Z_placed_large.model'
# base_data = local_path + '/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/wOOQ/wst2dry-wOOQPeriods_lastIter.dat'
# base_list = local_path + '/data/Regions/MetalEarth/dryden/j2/wst2dry_wOOQ_cull.lst'
# data_file = local_path + '/data/Regions/MetalEarth/dryden/dryOnly_wOOQ/dryOnly_wOOQ_base.dat'
# list_file = local_path + '/data/Regions/MetalEarth/dryden/j2/dryOnly_wOOQ.lst'
# model_out = local_path + '/data/Regions/MetalEarth/dryden/dryOnly_wOOQ/dry_all_placed_nest.model'
# data_out = local_path + '/data/Regions/MetalEarth/dryden/dryOnly_wOOQ/dry_all_placed.data'
########################################
# LIBEREC
# data_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/Liberec/4site/4site_reg.dat'
# base_data = 'C:/Users/eroots/phd/Nextcloud/data/Regions/Liberec/2site/901-902_reg.dat'
# list_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/Liberec/j2/allsites.lst'
# base_list = 'C:/Users/eroots/phd/Nextcloud/data/Regions/Liberec/j2/2site.lst'
# model_file = 'C:/Users/eroots/phd/Nextcloud/data/Regions/Liberec/2site/smoother/lib_NLCG_008.rho'

# model_out = r'C:/Users\eroots\phd\Nextcloud\data\Regions\Liberec/4site/4site_placed.model'
# data_out = r'C:\Users\eroots\phd\Nextcloud\data\Regions\Liberec/4site/4site_placed.data'
produce_large_model = 0
plot_it = 0
write_it = 1
plot_depth = 20000  # in meters
azi = -0  # Rotate the non-base data back

mod = WSDS.Model(model_file)
data = WSDS.Data(datafile=data_file, listfile=list_file)
data.locations = data.get_locs(azi=-azi)
data.set_locs()
base_data = WSDS.Data(datafile=base_data, listfile=base_list)
for site in base_data.site_names:
    if site in data.site_names:
        x_diff = data.sites[site].locations['X'] - base_data.sites[site].locations['X']
        y_diff = data.sites[site].locations['Y'] - base_data.sites[site].locations['Y']
        break
data.locations[:, 0] -= x_diff
data.locations[:, 1] -= y_diff
file_format = 'wsinv3dmt'

if produce_large_model:
    mod_large = WSDS.Model(model_file_large)
    x_large, y_large = [], []
    dx_orig = deepcopy(mod.dx)
    dy_orig = deepcopy(mod.dy)
    for ii, xL in enumerate(mod_large.dx):
        # xL = (mod_large.dx[ii] + mod_large.dx[ii + 1]) / 2
        if xL < np.min(dx_orig):
            mod.dx_insert(ii, mod_large.dx[ii])
        elif xL > np.max(dx_orig):
            mod.dx_insert(mod.nx + 1, mod_large.dx[ii])
            # x_include.append(ii)
    for ii, yL in enumerate(mod_large.dy):
        # yL = (mod_large.dy[ii] + mod_large.dy[ii + 1]) / 2
        if yL < np.min(dy_orig):
            mod.dy_insert(ii, mod_large.dy[ii])
        elif yL > np.max(dy_orig):
            mod.dy_insert(mod.ny + 1, mod_large.dy[ii])
    for ix, xx in enumerate(utils.edge2center(mod.dx)):
        for iy, yy in enumerate(utils.edge2center(mod.dy)):
            if xx < np.min(dx_orig) or xx > np.max(dx_orig) or yy < np.min(dy_orig) or yy > np.max(dy_orig):
                idx = np.argmin(abs(xx - np.array(mod_large.dx)))
                idy = np.argmin(abs(yy - np.array(mod_large.dy)))
                mod.vals[ix, iy, :] = mod_large.vals[idx, idy, :]

x, y, z = (utils.edge2center(arr) for arr in (mod.dx, mod.dy, mod.dz))
for ii in range(len(mod.dx) - 1):
    x[ii] = (mod.dx[ii] + mod.dx[ii + 1]) / 2
for ii in range(len(mod.dy) - 1):
    y[ii] = (mod.dy[ii] + mod.dy[ii + 1]) / 2
for ii in range(len(mod.dz) - 1):
    z[ii] = (mod.dz[ii] + mod.dz[ii + 1]) / 2

            # y_include.append(ii)
    # for ii in range(len(mod_large.dz) - 1):
    #     zL = (mod_large.dz[ii] + mod_large.dz[ii + 1]) / 2
    #     if zL < np.min(mod.dz) and xL > np.max(mod.dz):
    #         z.append(zL)
    #         z_include.append(ii)

# x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
# x_grid, y_grid, z_grid = (np.ravel(arr) for arr in (x_grid, y_grid, z_grid))
# X, Y = (x, y)
# bot_edge = -34000
# top_edge = 10000
# left_edge = -25000
# right_edge = 25000
# bot_edge = 0
# top_edge = 40000
# left_edge = -7000
# right_edge = 18000
# bot_edge = -230000
# top_edge = -50000
# left_edge = -200000
# right_edge = -15000
# x_interp = 60
# y_interp = 60
# n_xpad = 20
# n_ypad = 20
# x_pad_extention = 250000  # These control the total width of the combined padding
# y_pad_extention = 250000
# min_depth = 10
# max_depth = 500000
# #####################################
# # SWAYZE NORTH
# bot_edge = 13000
# top_edge = 26000
# left_edge = 7000
# right_edge = 13000
# x_interp = 75
# y_interp = 40
# n_xpad = 15
# n_ypad = 15
# x_pad_extention = 75000  # These control the total width of the combined padding
# y_pad_extention = 75000
# min_depth = 1
# max_depth = 100000
#####################################
# SWAYZE SOUTH-EAST
# bot_edge = -42000
# top_edge = -31500
# left_edge = 8000
# right_edge = 16000
# x_interp = 80
# y_interp = 75
# n_xpad = 25
# n_ypad = 25
# pad_multiplier = 1.25
# bot_edge = min(data.locations[:, 0]) - 1000
# top_edge = max(data.locations[:, 0]) + 1000
# left_edge = min(data.locations[:, 1]) - 1000
# right_edge = max(data.locations[:, 1]) + 1000
# x_interp = int(abs(bot_edge - top_edge) / 100)
# y_interp = int(abs(left_edge - right_edge) / 100)
# # x_interp = 100
# # y_interp = 70
# n_xpad = 10
# n_ypad = 10
# pad_multiplier = 1.2
# min_depth = 10
# max_depth = 1000000
#####################################
# DRYDEN CENTRAL
# bot_edge = 5250
# top_edge = 18500
# left_edge = -16000
# right_edge = -10500
# x_interp = 100
# y_interp = 70
# n_xpad = 25
# n_ypad = 25
# pad_multiplier = 1.2
####################################
# LIBEREC
# bot_edge = -500
# top_edge = 4000
# left_edge = -800
# right_edge = 500
# x_interp = 80
# y_interp = 40
# n_xpad = 13
# n_ypad = 17
# pad_multiplier = 1.2
# x_pad_extention = 75000  # These control the total width of the combined padding
# y_pad_extention = 75000
#####################################
# DRYDEN WST2DRY
# bot_edge = -225000
# top_edge = 50000
# left_edge = -250000
# right_edge = 0
# bot_edge = min(data.locations[:, 0]) + 35000
# top_edge = max(data.locations[:, 0]) - 20000
# left_edge = min(data.locations[:, 1]) + 40000
# right_edge = max(data.locations[:, 1]) - 15000
# x_interp = int(abs(bot_edge - top_edge) / 1500)
# y_interp = int(abs(left_edge - right_edge) / 1500)
# # x_interp = 100
# # y_interp = 70
# n_xpad = 15
# n_ypad = 15
# pad_multiplier = 1.1
# min_depth = 10
# max_depth = 1000000
######################################
# STURGEON WST2STU
bot_edge = min(data.locations[:, 0]) - 1500
top_edge = max(data.locations[:, 0]) + 1500
left_edge = min(data.locations[:, 1]) - 1500
right_edge = max(data.locations[:, 1]) + 1500
x_interp = int(abs(bot_edge - top_edge) / 2500)
y_interp = int(abs(left_edge - right_edge) / 2500)
# x_interp = 100
# y_interp = 70
n_xpad = 30
n_ypad = 30
# n_xpad = 25
# n_ypad = 25
pad_multiplier = 1.1
# n_xpad = 10
# n_ypad = 10
min_depth = 10
max_depth = 650000
depths_per_decade = (10, 12, 14, 10, 8)
dz = utils.generate_zmesh(min_depth=min_depth, max_depth=max_depth, NZ=depths_per_decade)[0]
dz[-3] = 410000
dz = mod.dz
fill_value = 1000
# max_depth = mod.dz[-1]
x_interior = np.linspace(bot_edge, top_edge, x_interp)
x_pad_size = (x_interior[-1] - x_interior[-2]) * 1.5
y_interior = np.linspace(left_edge, right_edge, y_interp)
y_pad_size = (y_interior[-1] - y_interior[-2]) * 1.5
left_pad, right_pad = [y_pad_size], [y_pad_size]
bot_pad, top_pad = [x_pad_size], [x_pad_size]
for ix in range(1, n_xpad):
    bot_pad.append(bot_pad[ix - 1] * pad_multiplier)
    top_pad.append(top_pad[ix - 1] * pad_multiplier)
bot_pad = np.flip(bot_edge - np.cumsum(np.array(bot_pad)), 0)
top_pad = top_edge + np.cumsum(np.array(top_pad))
for iy in range(1, n_ypad):
    left_pad.append(left_pad[iy - 1] * pad_multiplier)
    right_pad.append(right_pad[iy - 1] * pad_multiplier)
left_pad = np.flip(left_edge - np.cumsum(np.array(left_pad)), 0)
right_pad = right_edge + np.cumsum(np.array(right_pad))
# top_pad = np.logspace(np.log10(x_pad_size),
#                       np.log10(x_pad_extention), n_xpad) + top_edge
# # bot_pad = np.flip(-1 * (np.logspace(np.log10(x_pad_size),
#                                     # np.log10(x_pad_extention), n_xpad) + (bot_edge)), 0)
# # bot_pad = -1 * np.flip((np.logspace(np.log10(x_pad_size),
#                                     # np.log10(x_pad_extention), n_xpad) - (bot_edge)), 0)
# # left_pad = np.flip(-1 * (np.logspace(np.log10(y_pad_size),
#                                      # np.log10(y_pad_extention), n_ypad) + (left_edge)), 0)
# # left_pad = np.logspace(np.log10(y_pad_size), abs(np.log10(left_edge - y_pad_extention)), n_ypad)
# bot_pad = np.flip(bot_edge - np.logspace(np.log10(x_pad_size), np.log10(x_pad_extention), n_xpad), 0)
# left_pad = np.flip(left_edge - np.logspace(np.log10(y_pad_size), np.log10(y_pad_extention), n_ypad), 0)
# # left_pad = -1 * np.flip((np.logspace(np.log10(y_pad_size),
#                                      # np.log10(y_pad_extention), n_ypad) - abs(left_edge)), 0)
# right_pad = np.logspace(np.log10(y_pad_size),
#                         np.log10(y_pad_extention), n_ypad) + right_edge
x_mesh = (np.concatenate((bot_pad, x_interior, top_pad)))
y_mesh = (np.concatenate((left_pad, y_interior, right_pad)))
X = utils.edge2center(x_mesh)
Y = utils.edge2center(y_mesh)
# dz, CSZ, ddz = utils.generate_zmesh(min_depth=min_depth, max_depth=max_depth, NZ=depths_per_decade)
# dz = mod.dz
Z = utils.edge2center(dz)

X_grid, Y_grid, Z_grid = np.meshgrid(X, Y, Z)
X_grid, Y_grid, Z_grid = (np.ravel(arr) for arr in (X_grid, Y_grid, Z_grid))
# interp = griddata((x_grid, y_grid, z_grid),
#                   np.ravel(mod.vals),
#                   (X_grid, Y_grid, Z_grid),
#                   method='nearest')
interp = RGI((y, x, z),
             np.transpose(mod.vals, [1, 0, 2]),
             method='nearest', bounds_error=False, fill_value=fill_value)
query_points = np.array((Y_grid, X_grid, Z_grid)).T
new_vals = interp(query_points)
new_vals = np.reshape(new_vals, [len(Y), len(X), len(Z)])
new_vals = np.transpose(new_vals, [1, 0, 2])
ii = np.argmin(abs(np.array(dz[1:]) - mod.dz[1]))
new_vals[:, :, 0:ii] = np.transpose(np.tile(new_vals[:, :, ii], [ii, 1, 1]), [1, 2, 0])
# vals = interp()
if write_it:
    mod.vals = new_vals
    mod.dx, mod.dy, mod.dz = (x_mesh, y_mesh, dz)
    mod.write(model_out, file_format=file_format)
    center = mod.center
    data.locations[:, 0] -= center[0]
    data.locations[:, 1] -= center[1]
    data.write(outfile=data_out, file_format=file_format)
if plot_it:
    zz_1 = np.argmin(abs(plot_depth - np.array(mod.dz)))
    zz_2 = np.argmin(abs(plot_depth - np.array(dz)))
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].pcolormesh(y, x, np.log10(mod.vals[:, :, zz_1]),
                       cmap=cm.jet_plus(64), vmin=1, vmax=5,
                       edgecolor='k', linewidth=0.01)
    axes[0].plot(base_data.locations[:, 1], base_data.locations[:, 0], 'kv')
    axes[0].plot(data.locations[:, 1], data.locations[:, 0], 'w^')
    axes[1].pcolormesh(Y, X, np.log10(new_vals[:, :, zz_2]),
                       cmap=cm.jet_plus(64), vmin=1, vmax=5,
                       edgecolor='k', linewidth=0.01)
    axes[1].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    # axes[0].invert_yaxis()
    # axes[1].invert_yaxis()
    # fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    # axes2[0].pcolormesh(y, x, np.log10(mod.vals[:, :, 36]),
    #                     cmap=cm.jet_plus(64), vmin=1, vmax=5)
    # axes[0].plot(base_data.locations[:, 1], base_data.locations[:, 0], 'kv')
    # axes2[0].plot(data.locations[:, 1], data.locations[:, 0], 'w^')
    # axes2[1].pcolormesh(Y, X, np.log10(new_vals[:, :, 44]),
    #                     cmap=cm.jet_plus(64), vmin=1, vmax=5)
    # axes2[1].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    # axes2[0].invert_yaxis()
    # axes2[1].invert_yaxis()
    plt.show()
