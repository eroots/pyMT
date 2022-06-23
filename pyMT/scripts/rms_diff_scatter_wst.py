import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from pyMT.e_colours import colourmaps as cm
import naturalneighbor as nn


local_path = 'E:'
file_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots/feature_tests/'

listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
base_response = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/wstZK_resp.dat'
response_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/'
response_file = ['wstZK-C2_resistor_resp.dat', 'wstZK_C5a-10000ohm_NLCG_000.dat',
                 'wstZK-C2_500ohm_resp.dat', 'wstZK_depthTest_NLCG_000.dat']
tags = ['C2_resistor', 'C5a-north', 'C2_500ohm', '100km_depth']

# data = WSDS.Data(datafile=datafile)
# data = WSDS.RawData(listfile)
raw_data = WSDS.RawData(listfile)
raw_data.locations = raw_data.get_locs(mode='lambert')
for ii, site in enumerate(raw_data.site_names):
    raw_data.sites[site].locations['X'] = raw_data.locations[ii, 0]
    raw_data.sites[site].locations['Y'] = raw_data.locations[ii, 1]

save_fig = 1
n_interp = 150
dpi = 300
padding = 30
use_rms_difference = 1
# interp_type = ('nn', 'linear', 'cubic')
# interp_type = ['nnscatter']
annotate_sites = 0
interp_type = ['scatter']
# cmap = cm.get_cmap('turbo', 5)
cmap = cm.get_cmap('hot_r', 16)
# cmap = cm.get_cmap('coolwarm', N=16)
use_cax = [0., 4]
base_data = WSDS.Data(datafile=data_file, listfile=listfile)
base_response = WSDS.Data(datafile=base_response, listfile=listfile)
base_dataset = WSDS.Dataset(listfile=raw_data)
base_dataset.data = base_data
base_dataset.response = base_response
base_rms = base_dataset.calculate_RMS()
for ii, resp in enumerate(response_path):
    file_name = tags[ii]
    response = WSDS.Data(datafile=response_path + response_file[ii], listfile=listfile)
    dataset = WSDS.Dataset(listfile=raw_data)
    dataset.data = base_data
    dataset.response = response
    rms = dataset.calculate_RMS()
    if use_rms_difference:
        rms_diff = {}
        for site in base_dataset.data.site_names:
            rms_diff.update({site: rms['Station'][site]['Total'] - base_rms['Station'][site]['Total']})
        rms = rms_diff
    else:
        rms = {site: rms['Station'][site]['Total'] for site in data.site_names}
    total_rms = np.sqrt(np.mean([r ** 2 for r in rms.values()]))

    for interp in interp_type:
        # file_name = 'swzPTScatter-southConduit_RMS_{}'.format(interp)
        periods = []
        loc = []
        rho2 = []
        vals = []
        z_loc = []
        if interp == 'nn':
            dims = [0, 1]
        else:
            dims = [0]
        for ss, site in enumerate(base_dataset.site_names):
            # for ii, p in enumerate(data.sites[site].periods):
                # if p in data.narrow_periods.keys():
                # if p > 0.01 and p < 1500:
                for dim in dims:
                # if data.narrow_periods[p] > 0.9:
                    periods.append(base_dataset.raw_data.sites[site].locations['X'] / 1000)
                    loc.append(base_dataset.raw_data.sites[site].locations['Y'] / 1000)
                    z_loc.append(dim)
                    vals.append(rms[site])
                    # if base_data:
                        # rho_diff_vals.append(rhodiff[site][ii])
                        # phase_diff_vals.append(phadiff[site][ii])
                        # diffvals.append(diff[site][ii])
        periods = np.array(periods)
        # periods = np.log10(periods)
        vals = np.array(vals)
        locs = np.array(loc)
        if 'nn' in interp:
            points = np.transpose(np.array((locs, periods, z_loc)))
        elif interp != 'scatter':
            points = np.transpose(np.array((locs, periods)))

        min_x, max_x = (min(loc), max(loc))
        min_p, max_p = (min(periods), max(periods))
        grid_ranges = [[min_x, max_x, n_interp * 1j],
                       [min_p, max_p, n_interp * 1j],
                       [0, 1, 1]]

        grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp), np.linspace(min_p, max_p, n_interp))
        if 'nn' in interp:
            grid_vals = np.squeeze(nn.griddata(points, vals, grid_ranges)).T
        elif interp != 'scatter':
            grid_vals = griddata(points, vals, (grid_x, grid_y), method=interp)
        plt.figure(figsize=(12, 8))
        if interp != 'scatter':
            plt.pcolor(grid_x, grid_y, grid_vals, cmap=cmap, vmin=use_cax[0], vmax=use_cax[1])
        if 'scatter' in interp:
            plt.scatter(locs, periods, s=60, c=vals,
                        cmap=cmap, vmin=use_cax[0], vmax=use_cax[1],
                        edgecolors='k', linewidths=1)
            if annotate_sites:
                for ii, txt in enumerate(vals):
                    plt.text(s='{:3.2f}'.format(vals[ii]), x=locs[ii], y=periods[ii])
        # plt.plot(locs, periods, 'kx')
        plt.gca().set_xlabel('Easting (km)')
        plt.gca().set_ylabel('Northing (km)')
        plt.gca().set_xlim([min_x - padding, max_x + padding])
        plt.gca().set_ylim([min_p - padding, max_p + padding])
        plt.gca().set_aspect(1)
        plt.gca().set_title('Total RMS: {:>4.2f}'.format(total_rms))
        cb = plt.colorbar()
        cb.set_label('RMS Misfit',
                     rotation=270,
                     labelpad=20,
                     fontsize=16)
        plt.clim(use_cax)
        if save_fig:
            # tag = '_{}{}'.format(dtype, comp)
            for ext in ['.svg', '.png']:
                plt.gcf().savefig(file_path + file_name + ext, dpi=dpi, bbox_inches='tight')
            plt.close()
if not save_fig:                
    plt.show()
