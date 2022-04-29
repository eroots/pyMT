import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from pyMT.e_colours import colourmaps as cm
import naturalneighbor as nn


local_path = 'E:'
# listfile = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/edi/all.lst'
# datafile = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/tests/all_inv18.dat'
# file_path = 'E:/phd/NextCloud/Documents/GoldenTriangle/RoughFigures/rms_plots/'
# file_name = 'line3_scatter-all'
file_path = 'E:/phd/NextCloud/Documents/ME_Transects/wst/rms_plots/'
file_name = 'rms_scatterplot'
# listfile = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/line3_plus.lst'
# base_data_file = local_path + '/phd/Nextcloud/data/Regions/snorcle/line3_plus/line3_wTopo_all_removed.dat'
# datafile = local_path + '/phd/NextCloud/data/Regions/snorcle/line3_plus/line3-all_lastIter.dat'
listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
base_data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat'
datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.dat'

# data = WSDS.Data(datafile=datafile)
# data = WSDS.RawData(listfile)
raw_data = WSDS.RawData(listfile)
raw_data.locations = raw_data.get_locs(mode='lambert')
for ii, site in enumerate(raw_data.site_names):
    raw_data.sites[site].locations['X'] = raw_data.locations[ii, 0]
    raw_data.sites[site].locations['Y'] = raw_data.locations[ii, 1]

n_interp = 150
save_fig = 1
# file_name = 'swzFinish_resp'
# file_name = 'swz_1000ohm_resp'
dpi = 300
padding = 30
# interp_type = ('nn', 'linear', 'cubic')
# interp_type = ['nnscatter']
annotate_sites = 0
interp_type = ['scatter']
# cmap = cm.get_cmap('turbo', 5)
cmap = cm.get_cmap('bgy_r', 7)
# cmap = cm.get_cmap('coolwarm', N=16)
use_cax = [0.5, 4]
# use_cax = [-4, 4]
# depths = ['5.0', '10.0', '14.0', '23.0']
# rhos = [10, 50, 100, 300, 500]
# depths = [500, 1000, 1500, 2500, 3500, 5000, 7000, 10000, 15000, 20000, 25000, 35000]
depths = [0]
rhos = [100]
for d in depths:
    for r in rhos:
        # file_path = 'E:/phd/NextCloud/Documents/ME_Transects/Swayze_paper/RoughFigures/featureTests/'.format(r)
        # base_data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/swz_cull1M_TFPT_regErrs.dat'
        # base_data_file = 'E:/phd/NextCloud/data/Regions/TTZ/full_run/ZK/1D/ttz_fullZK_flagged.dat'
        # base_data_file = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/tests/all_inv18.dat'
        # datafile = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/plc18_NLCG_116.dat'
        # datafile2 = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/tests/plc_rotatedTest-{}m_resp.dat'.format(d)
        # base_data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swz_cull1M_all.dat'
        # datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swzFinish_lastIter.dat'
        # datafile2 = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/swzPT_southConduitTest_resp.dat'
        # datafile2 = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/north_conduit_test/swzFT1_resp.dat'
        # datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/swzPT_lastIter_resp.dat'
        # datafile2 = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/{}ohm/swz_lccTest_{}ohm_{}kmDepth_resp.dat'.format(r, r, d)
        # base_data_file = local_path + '/phd/Nextcloud/data/Regions/snorcle/cull1b/hs100/sno_cull1b_K_reErredL3.dat'
        # datafile = local_path + '/phd/Nextcloud/data/Regions/snorcle/cull1b/hs100/sno-hs100-K_lastIter.dat'
        # base_data_file = local_path + '/phd/Nextcloud/data/Regions/snorcle/line2a_plus/line2a_wTopo_K_erred_removed2.dat'
        # datafile = local_path + '/phd/Nextcloud/data/Regions/snorcle/line2a_plus/line2a-0p2_K-fromZ_lastIter.dat'
        # data_files = [datafile, datafile2]
        data_files = [datafile]
        data = WSDS.Data(datafile=datafile, listfile=listfile)
        base_data = WSDS.Data(datafile=base_data_file, listfile=listfile)
        pt_comps = ('PTXX', 'PTYY', 'PTXY', 'PTYX')
        rms = []
        for file in data_files:
            data = WSDS.Data(datafile=file, listfile=listfile)
            # for site in data.site_names:
                # for comp in pt_comps:
                #     data.sites[site].data[comp] = np.array([getattr(pt, comp) for pt in data.sites[site].phase_tensors])
                # data.sites[site].components = data.PHASE_TENSOR_COMPONENTS + data.TIPPER_COMPONENTS
            # data.components = data.PHASE_TENSOR_COMPONENTS + data.TIPPER_COMPONENTS
            dataset = WSDS.Dataset(listfile=raw_data)
            dataset.data = base_data
            dataset.response = data
            # dataset.data.components = data.IMPEDANCE_COMPONENTS
            # dataset.response.components = data.IMPEDANCE_COMPONENTS
            rms.append(dataset.calculate_RMS())
        if len(rms) > 1:
            rms_diff = {}
            for site in data.site_names:
                rms_diff.update({site: rms[1]['Station'][site]['Total'] - rms[0]['Station'][site]['Total']})
            rms = rms_diff
        else:
            rms = {site: rms[0]['Station'][site]['Total'] for site in data.site_names}
        total_rms = np.sqrt(np.mean([r ** 2 for r in rms.values()]))
        # file_name = 'swzBase_lineardet'.format(r, d)
        # file_path = 'E:/phd/NextCloud/Documents/ME_Transects/Swayze_paper/RoughFigures/featureTests/{}ohm/'.format(r)
        # file_path = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/tests/rms_plots/'
        # file_path = 'E:/phd/NextCloud/Documents/GoldenTriangle/RoughFigures/rms_plots/'
        # file_name = 'line2a_scatter-K'.format(d)
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
            for ss, site in enumerate(data.site_names):
                # for ii, p in enumerate(data.sites[site].periods):
                    # if p in data.narrow_periods.keys():
                    # if p > 0.01 and p < 1500:
                    for dim in dims:
                    # if data.narrow_periods[p] > 0.9:
                        periods.append(dataset.raw_data.sites[site].locations['X'] / 1000)
                        loc.append(dataset.raw_data.sites[site].locations['Y'] / 1000)
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
                plt.scatter(locs, periods, s=100, c=vals,
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
                ext = '.svg'
                plt.gcf().savefig(file_path + file_name + ext, dpi=dpi, bbox_inches='tight')
                plt.close()
    if not save_fig:                
        plt.show()
