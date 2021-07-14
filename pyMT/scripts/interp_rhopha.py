import pyMT.data_structures as WSDS
import pyMT.utils as utils
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
from pyMT.e_colours import colourmaps as cm
import naturalneighbor as nn


# listfile = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/dbr15/j2/allsites.lst'
# datafile = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Test_Models/dimensionality/synthLayer.data'
# listfile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/main_transect.lst'
# datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/PT/lcc_test/10/swz_LCC_1000ohm_resp.dat'
# datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/lcc_test/1000/swz_LCC_1000ohm_resp.dat'
# datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/lcc_test/shallower/swz_LCC_shallower100ohm_resp.dat'
# datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swzFinish_lastIter.dat'
# datafile = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swz_cull1M_all.dat'
# listfile = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/edi/all_sorted.lst'
out_path = 'E:/phd/Nextcloud/Documents/ME_Transects/Upper_Abitibi/Paper/RoughFigures/feature_tests/'
base_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/AG/Hex2Mod/MC-tests/'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/j2/ROU-MAL_BB.lst'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/j2/LAR-ROU_BB.lst'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/matheson/j2/MATBB.lst'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/lar_bb.lst'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/rouyn/j2/ROUall.lst'
# list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/malartic/j2/MALBB.lst'
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/AG/Hex2Mod/MC-tests/AG_depthTest_baseCase_resp.dat'
base_data = WSDS.Data(data_file)
# data = WSDS.Data(datafile=datafile)
# data = WSDS.RawData(listfile)
raw_data = WSDS.RawData(list_file)


n_interp = raw_data.NS *2
save_fig = 1
pad = 0.5
dpi = 300
x_axis = 'station'
# interp_type = ('nn', 'linear', 'cubic')
interp_type = ['nearest']
# to_plot = ['rho', 'phase', 'diff']
to_plot = ['phasediff']
# to_plot = ['phase', 'phasediff']

# to_plot = ['phase']
component = ['xy', 'yx']
# component = ('xy', 'yx', 'det')
# rmsites = [site for site in data.site_names if site[0] == 'e' or site[0] == 'd']
# rmsites = [site for site in data.site_names if site not in raw_data.site_names]
# data.remove_sites(sites=rmsites)
# phase_cax = [30, 75]
# rho_cax = [0, 5]
# # data.sort_sites(order='south-north')
# idx = data.locations[:, 0].argsort()
# data.locations = data.locations[idx]  # Make sure they go north-south
# data.site_names = [data.site_names[ii] for ii in idx]
# A little kludge to make sure the last few sites are in the right order (west-east)
# data.locations[0:8, :] = data.locations[np.flip(data.locations[0:8, 1].argsort(), 0)]

# depths = ['5.0', '10.0', '14.0', '23.0']
# rhos = [10, 50, 100, 300, 500]
# depths = [500]
# depths = [500, 1000, 1500, 2500, 3500, 5000, 7000, 10000, 15000, 20000, 25000, 35000]
# rhos = [500]
# for d in depths:
    # for r in rhos:
for num in [2]:
    if True:
        tag = 'R{}'.format(num)
        resp_file = base_path + '{}/{}-300ohm_resp.dat'.format(tag, tag)
        # resp_file = base_path + '{}/{}-1000ohm-refined_resp.dat'.format(tag, tag)
        data = WSDS.Data(datafile=resp_file)
        # base_data = []
        # base_data = WSDS.Data(datafile=base_data_file)
        # Reorder based on list file
        # base_data.site_names = raw_data.site_names
        # data.site_names = raw_data.site_names
        
        rmsites = [site for site in data.site_names if site not in raw_data.site_names]
        data.remove_sites(sites=rmsites)
        if base_data:
            base_data.remove_sites(sites=rmsites)
        phase_cax = [30, 80]
        rho_cax = [0, 5]
        rhodiff_cax = [-1, 1]
        phasediff_cax = [-15, 15]
        # data.sort_sites(order='south-north')
        idx = data.locations[:, 1].argsort()
        data.locations = data.locations[idx]  # Make sure they go north-south
        data.site_names = [data.site_names[ii] for ii in idx]
        print('data loaded')
        for comp in component:
            out_file = '{}_pha{}'.format(tag, comp)
            # print(comp)
            rho = {site: utils.compute_rho(data.sites[site], calc_comp=comp)[0] for site in data.site_names}
            pha = {site: utils.compute_phase(data.sites[site], calc_comp=comp, wrap=1)[0] for site in data.site_names}
            
            if base_data:
                # if 'rho' in comp:
                pha_errs = {site: (utils.compute_phase(base_data.sites[site], calc_comp='xy', wrap=1, errtype='used_error')[1] +
                               utils.compute_phase(base_data.sites[site], calc_comp='yx', wrap=1, errtype='used_error')[1]) / 2 for site in data.site_names}
                base = {site: utils.compute_rho(base_data.sites[site], calc_comp=comp)[0] for site in base_data.site_names}
                rhodiff = {site: (rho[site] - base[site]) for site in base_data.site_names}
                # elif 'phase' in comp:
                base = {site: utils.compute_phase(base_data.sites[site], calc_comp=comp, wrap=1)[0] for site in base_data.site_names}
                phadiff = {site: (pha[site] - base[site]) for site in base_data.site_names}
            else:
                pha_errs = {site: np.zeros(data.NP) for site in data.site_names}
            # bost = {site.name: utils.compute_bost1D(site, calc_comp=comp)[0] for site in data.sites.values()}
            # depths = {site.name: utils.compute_bost1D(site)[1] for site in data.sites.values()}
            dist = utils.linear_distance(data.locations[:, 1], data.locations[:,0]) / 1000
            for interp in interp_type:
                print('Performing interp {}'.format(interp))
                base_periods, periods = [], []
                base_loc, loc = [], []
                rho2 = []
                phavals = []
                rho_diff_vals, phase_diff_vals, base_phase_vals = [], [], []
                depths2 = []
                bost2 = []
                base_z_loc, z_loc = [], []
                if interp == 'nn':
                    dims = [0, 1]
                else:
                    dims = [0]
                for ss, site in enumerate(data.site_names):
                    for ii, p in enumerate(data.sites[site].periods):
                        # if p in data.narrow_periods.keys():
                        # if p > 0.01 and p < 1500:
                        for dim in dims:
                        # if data.narrow_periods[p] > 0.9:
                            periods.append(p)
                            # bost2.append(bost[site][ii])
                            # depths2.append(depths[site][ii])
                            rho2.append(rho[site][ii])
                            phavals.append(pha[site][ii])
                            # loc.append(data.sites[site].locations['X'])
                            if x_axis == 'linear':
                                loc.append(dist[ss])
                            elif x_axis == 'station':
                                loc.append(ss)
                            z_loc.append(dim)
                            if base_data:
                                # if pha_errs[site][ii] < 5:
                                if x_axis == 'linear':
                                    base_loc.append(dist[ss])
                                elif x_axis == 'station':
                                    base_loc.append(ss)
                                base_periods.append(p)
                                rho_diff_vals.append(rhodiff[site][ii])
                                phase_diff_vals.append(phadiff[site][ii])
                                base_phase_vals.append(base[site][ii])
                                base_z_loc.append(dim)
                                # diffvals.append(diff[site][ii])
                phavals = np.array(phavals)
                rho2 = np.array(rho2)
                # depths2 = np.array(depths2)
                # bost2 = np.array(bost2)
                rhovals = np.log10(rho2)
                # bostvals = np.log10(bost2)
                periods = np.array(periods)
                periods = np.log10(periods)
                base_phase_vals = np.array(base_phase_vals)
                base_periods = np.array(base_periods)
                base_periods = np.log10(base_periods)
                base_locs = np.array(base_loc)
                base_z_loc = np.array(base_z_loc)
                rho_diff_vals = np.array(rho_diff_vals)
                phase_diff_vals = np.array(phase_diff_vals)
                locs = np.array(loc)
                if interp == 'nn':
                    points = np.transpose(np.array((locs, periods, z_loc)))
                    if base_data:
                        base_points = np.transpose(np.array((base_locs, base_periods, base_z_loc)))
                else:
                    points = np.transpose(np.array((locs, periods)))
                    if base_data:
                        base_points = np.transpose(np.array((base_locs, base_periods)))
                min_x, max_x = (min(loc) - pad, max(loc) + pad)
                min_p, max_p = (min(periods), max(periods))

                grid_ranges = [[min_x, max_x, n_interp * 1j],
                               [min_p, max_p, n_interp * 1j],
                               [0, 1, 1]]
                for dtype in to_plot:
                    print(dtype)
                    if dtype == 'phase':
                        use_cax = phase_cax
                        cmap = cm.get_cmap('turbo_capped', 32)
                    elif dtype == 'rho':
                        use_cax = rho_cax
                        cmap = cm.get_cmap('turbo_r', 32)
                    elif dtype == 'rhodiff':
                        use_cax = rhodiff_cax
                        cmap = cm.get_cmap('bwr', 32)
                    elif dtype == 'phasediff':
                        use_cax = phasediff_cax
                        cmap = cm.get_cmap('twilight_shifted', 32)
                    grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp), np.log10(np.logspace(min_p, max_p, n_interp)))
                    if dtype == 'rho':
                        vals = rhovals
                    elif 'phase' in dtype:
                        vals = phavals
                    elif 'rhodiff' in dtype:
                        vals = rho_diff_vals
                    elif 'phasediff' in dtype:
                        vals = phase_diff_vals
                        
                    if interp == 'nn':
                        print('in nn')
                        grid_vals = np.squeeze(nn.griddata(points, vals, grid_ranges)).T
                    else:
                        grid_vals = griddata(points, vals, (grid_x, grid_y), method=interp)
                    if 'diff' in dtype:
                        if interp == 'nn':
                            print('in nn')
                            base_vals = np.squeeze(nn.griddata(base_points, base_phase_vals, grid_ranges)).T
                        else:
                            base_vals = griddata(base_points, base_phase_vals, (grid_x, grid_y), method=interp)
                        grid_vals = grid_vals - base_vals
                    print('Plotting Figure')
                    plt.figure(figsize=(16, 12))
                    plt.pcolor(grid_x, grid_y, grid_vals, cmap=cmap, vmin=use_cax[0], vmax=use_cax[1])
                    
                    # plt.scatter(locs, periods, c=vals, cmap=cmap, vmin=0, vmax=5)
                    plt.gca().invert_yaxis()
                    plt.gca().set_xlabel('Distance (km)')
                    plt.gca().set_ylabel('Period (s)')
                    # plt.gca().set_title('{} {}, {}m'.format(comp, dtype, d))
                    if x_axis == 'station':
                        plt.xticks(range(data.NS), data.site_names, rotation='vertical')
                        plt.gca().set_xlabel('Station')
                    plt.colorbar()
                    plt.clim(use_cax)
                    # plt.contour(grid_x, grid_y, grid_vals, levels=[-3.5, 3.5], colors='k')
                    print('Done Plotting')
                    if save_fig:
                        # tag = '_{}'.format(dtype)
                        ext = '.png'
                        plt.gcf().savefig(out_path + out_file + ext, dpi=dpi, transparent=True)
                        plt.close()
                    else:          
                        print('Showing Figure')      
                        plt.show()
