import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from pyMT.e_colours import colourmaps as cm
import naturalneighbor as nn


local_path = 'E:'
# file_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots/feature_tests/perc_diff/'
file_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst-mantle/rms_plots/feature_tests/depth_tests/impXY/'
# file_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst-mantle/rms_plots/feature_tests/'

listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
base_response = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/anisotropic/wstZK_ani_lastIter.dat'
# response_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/feature_tests/shallower/'
# response_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/anisotropic/feature_tests/geotherms/western_edge/'
response_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/anisotropic/depth_tests/X/'

# response_file = ['wstZK_C2R-300ohm_resp.dat', 
#                  'wstZK_C2R-500ohm_resp.dat', 
#                  'wstZK_C2R-1000ohm_resp.dat', 
#                  'wstZK_C2_500ohm_resp.dat',
#                  'wstZK_C2_1000ohm_resp.dat',
#                  'wstZK_C2_3000ohm_resp.dat',
#                  'wstZK_C3-500ohm_resp.dat',
#                  'wstZK_C3-1000ohm_resp.dat',
#                  'wstZK_C3-3000ohm_resp.dat',
#                  'wstZK_C3R-South-300ohm_resp.dat',
#                  'wstZK_C3R-South-500ohm_resp.dat',
#                  'wstZK_C3R-South-1000ohm_resp.dat',
#                  'wstZK_C3R-North-300ohm_resp.dat',
#                  'wstZK_C3R-North-500ohm_resp.dat',
#                  'wstZK_C3R-North-1000ohm_resp.dat']
# response_file = ['wstZK_C2-3000ohm_resp.dat',
#                  'wstZK_C3-3000ohm_resp.dat',
#                  'wstZK_C2R-300ohm_resp.dat', 
#                  'wstZK_C3R-South-300ohm_resp.dat',
#                  'wstZK_C3R-North-300ohm_resp.dat',]
# response_file = ['wst_ani_R1-300ohm_56-147km_resp.dat',
#                  'wst_ani_R1-300ohm_71-147km_resp.dat',
#                  'wst_ani_R1-300ohm_89-147km_resp.dat']
# response_file = ['wst_aniY_hasterok33_resp.dat',
#                  'wst_aniY_hasterok34_resp.dat',
#                  'wst_aniY_hasterok35_resp.dat',
#                  'wst_aniY_hasterok36_resp.dat',
#                  'wst_aniY_hasterok37_resp.dat',
#                  'wst_aniY_hasterok38_resp.dat',
#                  'wst_aniY_hasterok39_resp.dat',
#                  'wst_aniY_hasterok40_resp.dat']
# response_file = ['wst_ZK_100km_resp.dat',
#                  'wst_ZK_114km_resp.dat',
#                  'wst_ZK_129km_resp.dat',
#                  'wst_ZK_147km_resp.dat',
#                  'wst_ZK_167km_resp.dat',
#                  'wst_ZK_190km_resp.dat',
#                  'wst_ZK_215km_resp.dat',
#                  'wst_ZK_245km_resp.dat',
#                  'wst_ZK_278km_resp.dat',
#                  'wst_ZK_316km_resp.dat',
#                  'wst_ZK_359km_resp.dat'
#                  ]
# tags = [x.strip('wstZK_ani_').strip('_resp.dat') for x in response_file]
response_file = ['wstZK_ani_100km_resp.dat',
                 'wstZK_ani_114km_resp.dat',
                 'wstZK_ani_129km_resp.dat',
                 'wstZK_ani_147km_resp.dat',
                 'wstZK_ani_167km_resp.dat',
                 'wstZK_ani_190km_resp.dat',
                 'wstZK_ani_215km_resp.dat',
                 'wstZK_ani_245km_resp.dat',
                 'wstZK_ani_278km_resp.dat',
                 'wstZK_ani_316km_resp.dat',
                 'wstZK_ani_359km_resp.dat'
                 ]
tags = [x.strip('wstZK_ani_').strip('_resp.dat') for x in response_file]
# tags = ['C3_resistor', 'C5a-north', 'C2_500ohm', '100km_depth']
# tags = [x.strip('wstZK_').strip('_resp.dat') for x in response_file]
# tags = [x.strip('wstZK_ani_').strip('_resp.dat') for x in response_file]
# tags = ['C2R-300ohm',
#         'C2R-500ohm',
#         'C2R-1000ohm',
#         'C2-500ohm',
#         'C2-1000ohm',
#         'C2-3000ohm',
#         'C3-500ohm',
#         'C3-1000ohm',
#         'C3-3000ohm',
#         'C3R-South-300ohm',
#         'C3R-South-500ohm',
#         'C3R-South-1000ohm',
#         'C3R-North-300ohm',
#         'C3R-North-500ohm',
#         'C3R-North-1000ohm']

# data = WSDS.Data(datafile=datafile)
# data = WSDS.RawData(listfile)
raw_data = WSDS.RawData(listfile)
raw_data.locations = raw_data.get_locs(mode='lambert')
for ii, site in enumerate(raw_data.site_names):
    raw_data.sites[site].locations['X'] = raw_data.locations[ii, 0]
    raw_data.sites[site].locations['Y'] = raw_data.locations[ii, 1]

# Clip out data outside of the bounding box
rm_sites = []
for site in raw_data.site_names:
    if -250000 < raw_data.sites[site].locations['X'] < 450000:
        continue
    else:
        rm_sites.append(site)
raw_data.remove_sites(sites=rm_sites)

save_fig = 1
n_interp = 150
dpi = 300
padding = 30
xlim = []
ylim = [-200, 450]
use_rms_difference  = 1 # If false, it shows the rms for the response itself
use_perc_difference = 0 # If False, use normal difference
use_component = 'XY'    # Choose a component, or 'all'. Not all implemented
# interp_type = ('nn', 'linear', 'cubic')
# interp_type = ['nnscatter']
annotate_sites = 0
interp_type = ['scatter']
# cmap = cm.get_cmap('turbo', 5)
# cmap = cm.get_cmap('hot_r', 10)
cmap = cm.get_cmap('bwr', N=17)
# If using a diverging map but non-symmetric endpoints, use this to re-center at 0
renormalize_cmap = False
cmap_center = 0

diff_cax = [-1, 1]
perc_cax = [0, 50]
if use_perc_difference:
    use_cax = perc_cax
else:
    use_cax = diff_cax
base_data = WSDS.Data(datafile=data_file, listfile=listfile)
base_data.remove_sites(sites=rm_sites)
base_response = WSDS.Data(datafile=base_response, listfile=listfile)
base_response.remove_sites(sites=rm_sites)
base_dataset = WSDS.Dataset(listfile=raw_data)
base_dataset.data = base_data
base_dataset.response = base_response
base_rms = base_dataset.calculate_RMS()

dataset = WSDS.Dataset(listfile=raw_data)
dataset.data = base_data

for ii, resp in enumerate(response_file):
    file_name = tags[ii]
    response = WSDS.Data(datafile=response_path + response_file[ii], listfile=listfile)
    response.remove_sites(sites=rm_sites)
    # dataset = WSDS.Dataset(listfile=raw_data)
    # dataset.data = base_data
    dataset.response = response
    rms = dataset.calculate_RMS()
    if use_rms_difference:
        rms_diff = {}
        for site in base_dataset.data.site_names:
            if use_perc_difference:
                if use_component == 'all':
                    rms_diff.update({site: 100 * (rms['Station'][site]['Total'] - base_rms['Station'][site]['Total']) / base_rms['Station'][site]['Total'] })
                elif use_component == 'YX':
                    rms_diff.update({site: 100 * (((rms['Station'][site]['ZYXR'] - base_rms['Station'][site]['ZYXR']) / base_rms['Station'][site]['ZYXR']) +
                                                   (rms['Station'][site]['ZYXR'] - base_rms['Station'][site]['ZYXR']) / base_rms['Station'][site]['ZYXR']) / 2})
            else:
                if use_component == 'all':
                    rms_diff.update({site: rms['Station'][site]['Total'] - base_rms['Station'][site]['Total']})
                elif use_component == 'XY':
                    rms_diff.update({site: (np.sqrt(np.mean([rms['Station'][site]['ZXYR']**2, rms['Station'][site]['ZXYI']**2])) - 
                                           (np.sqrt(np.mean([base_rms['Station'][site]['ZXYR']**2, base_rms['Station'][site]['ZXYI']]))))})
                elif use_component == 'YX':
                    rms_diff.update({site: (np.sqrt(np.mean([rms['Station'][site]['ZYXR']**2, rms['Station'][site]['ZYXI']**2])) - 
                                           (np.sqrt(np.mean([base_rms['Station'][site]['ZYXR']**2, base_rms['Station'][site]['ZYXI']]))))})
                elif use_component == 'phayx':
                    phase1, errors = utils.compute_phase(dataset.response.sites[site], calc_comp='YX', errtype='none', wrap=1)
                    phase2, errors = utils.compute_phase(base_dataset.response.sites[site], calc_comp='YX', errtype='none', wrap=1)
                    phase3, errors = utils.compute_phase(base_dataset.data.sites[site], calc_comp='YX', errtype='used_error', wrap=1)
                    rms1 = utils.rms((phase1 - phase3) / errors)
                    rms2 = utils.rms((phase2 - phase3) / errors)
                    rms_diff.update({site: rms1 - rms2})
        rms = rms_diff
    else:
        rms = {site: rms['Station'][site]['Total'] for site in data.site_names}
    total_rms = np.sqrt(np.mean([r ** 2 for r in rms.values() if r > 0]))

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
            if renormalize_cmap:
                plt.scatter(locs, periods, s=60, c=vals,
                            cmap=cmap, norm=TwoSlopeNorm(vcenter=cmap_center, vmin=use_cax[0], vmax=use_cax[1]),
                            edgecolors='k', linewidths=1)
            else:
                plt.scatter(locs, periods, s=60, c=vals,
                            cmap=cmap, vmin=use_cax[0], vmax=use_cax[1],
                            edgecolors='k', linewidths=1)
            if annotate_sites:
                for ii, txt in enumerate(vals):
                    plt.text(s='{:3.2f}'.format(vals[ii]), x=locs[ii], y=periods[ii])
        # plt.plot(locs, periods, 'kx')
        plt.gca().set_xlabel('Easting (km)')
        plt.gca().set_ylabel('Northing (km)')
        if xlim:
            plt.gca().set_xlim(xlim)
        else:
            plt.gca().set_xlim([min_x - padding, max_x + padding])
        if ylim:
            plt.gca().set_ylim(ylim)
        else:
            plt.gca().set_ylim([min_p - padding, max_p + padding])
        plt.gca().set_aspect(1)
    if use_rms_difference:
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
