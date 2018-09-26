import pyMT.data_structures as WSDS
from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
import pyMT.utils as utils
import matplotlib.pyplot as plt
import e_colours.colourmaps as cm


model_file = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.rho'
base_data = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\swz_cull1\finish\swz_finish.dat'
data_file = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2f_bb_Z.data'
list_file = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\j2\R1South_bb.lst'
base_list = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\j2\swz_cull1.lst'
plot_it = False
write_it = True
model_out = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\tmp.model'
data_out = r'C:\Users\eric\Documents\MATLAB\MATLAB\Inversion\Regions\MetalEarth\swayze\R1South_2\bb\R1South_2_bb_test.data'
mod = WSDS.Model(model_file)
data = WSDS.Data(datafile=data_file, listfile=list_file)
base_data = WSDS.Data(datafile=base_data, listfile=base_list)
for site in base_data.site_names:
    if site in data.site_names:
        x_diff = data.sites[site].locations['X'] - base_data.sites[site].locations['X']
        y_diff = data.sites[site].locations['Y'] - base_data.sites[site].locations['Y']
        break
data.locations[:, 0] -= x_diff
data.locations[:, 1] -= y_diff
file_format = 'wsinv3dmt'
depths_per_decade = (10, 12, 14, 16, 14, 6)
x, y, z = (utils.edge2center(arr) for arr in (mod.dx, mod.dy, mod.dz))
for ii in range(len(mod.dx) - 1):
    x[ii] = (mod.dx[ii] + mod.dx[ii + 1]) / 2
for ii in range(len(mod.dy) - 1):
    y[ii] = (mod.dy[ii] + mod.dy[ii + 1]) / 2
for ii in range(len(mod.dz) - 1):
    z[ii] = (mod.dz[ii] + mod.dz[ii + 1]) / 2
# x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
# x_grid, y_grid, z_grid = (np.ravel(arr) for arr in (x_grid, y_grid, z_grid))
# X, Y = (x, y)
bot_edge = -32000
top_edge = 0
left_edge = -5000
right_edge = 2000
x_interp = 100
y_interp = 50
n_xpad = 20
n_ypad = 20
x_pad_extention = 100000  # These control the total width of the combined padding
y_pad_extention = 100000
x_interior = np.linspace(bot_edge, top_edge, x_interp)
x_pad_size = (x_interior[-1] - x_interior[-2]) * 1.5
y_interior = np.linspace(left_edge, right_edge, y_interp)
y_pad_size = (y_interior[-1] - y_interior[-2]) * 1.5
top_pad = np.logspace(np.log10(x_pad_size),
                      np.log10(x_pad_extention), n_xpad) + top_edge
bot_pad = np.flip(-1 * (np.logspace(np.log10(x_pad_size),
                                    np.log10(x_pad_extention), n_xpad) + abs(bot_edge)), 0)
left_pad = np.flip(-1 * (np.logspace(np.log10(y_pad_size),
                                     np.log10(y_pad_extention), n_ypad) + abs(left_edge)), 0)
right_pad = np.logspace(np.log10(y_pad_size),
                        np.log10(y_pad_extention), n_ypad) + right_edge
x_mesh = (np.concatenate((bot_pad, x_interior, top_pad)))
y_mesh = (np.concatenate((left_pad, y_interior, right_pad)))
X = utils.edge2center(x_mesh)
Y = utils.edge2center(y_mesh)
dz, CSZ, ddz = utils.generate_zmesh(min_depth=1, max_depth=mod.dz[-1], NZ=depths_per_decade)
Z = utils.edge2center(dz)
X_grid, Y_grid, Z_grid = np.meshgrid(X, Y, Z)
X_grid, Y_grid, Z_grid = (np.ravel(arr) for arr in (X_grid, Y_grid, Z_grid))
# interp = griddata((x_grid, y_grid, z_grid),
#                   np.ravel(mod.vals),
#                   (X_grid, Y_grid, Z_grid),
#                   method='nearest')
interp = RGI((y, x, z),
             np.transpose(mod.vals, [1, 0, 2]),
             method='nearest', bounds_error=False, fill_value=2500)
query_points = np.array((Y_grid, X_grid, Z_grid)).T
new_vals = interp(query_points)
new_vals = np.reshape(new_vals, [len(Y), len(X), len(Z)])
new_vals = np.transpose(new_vals, [1, 0, 2])
# vals = interp()
if write_it:
    mod.vals = new_vals
    mod.dx, mod.dy, mod.dz = (x_mesh, y_mesh, dz)
    mod.write(model_out, file_format=file_format)
    center = mod.center
    # data.locations[:, 0] -= center[0]
    # data.locations[:, 1] -= center[1]
    data.write(outfile=data_out, file_format=file_format)
if plot_it:
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].pcolormesh(y, x, np.log10(mod.vals[:, :, 36]),
                       cmap=cm.jet_plus(64), vmin=1, vmax=5,
                       edgecolor='k', linewidth=0.01)
    axes[0].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    axes[1].pcolormesh(Y, X, np.log10(new_vals[:, :, 52]),
                       cmap=cm.jet_plus(64), vmin=1, vmax=5,
                       edgecolor='k', linewidth=0.01)
    axes[1].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    # axes[0].invert_yaxis()
    # axes[1].invert_yaxis()
    fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes2[0].pcolormesh(y, x, np.log10(mod.vals[:, :, 36]),
                        cmap=cm.jet_plus(64), vmin=1, vmax=5)
    axes2[0].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    axes2[1].pcolormesh(Y, X, np.log10(new_vals[:, :, 52]),
                        cmap=cm.jet_plus(64), vmin=1, vmax=5)
    axes2[1].plot(data.locations[:, 1], data.locations[:, 0], 'kv')
    # axes2[0].invert_yaxis()
    # axes2[1].invert_yaxis()
    plt.show()
