import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import numpy as np
import pyMT.utils as utils
import pyMT.gplot as gplot
import os
from copy import deepcopy


# local_path = 'C:/Users/eroots/'
local_path = 'E:'

def generate_ellipse(phi):
    step_size = np.pi / 30
    jx = np.cos(np.arange(0, 2 * np.pi + step_size, step_size))
    jy = np.sin(np.arange(0, 2 * np.pi + step_size, step_size))
    phi_x = phi[0, 0] * jx + phi[0, 1] * jy
    phi_y = phi[1, 0] * jx + phi[1, 1] * jy
    return phi_x, phi_y


###########################################
# DRYDEN
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/dry5_3.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/dryden/j2/main_transect_pt.lst'
###########################################
# MALARTIC
# listfile = local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_amt.lst'
# main_list = local_path + 'phd/ownCloud/data/Regions/MetalEarth/malartic/j2/mal_amt.lst'
###########################################
# LARDER
list_file = [local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb.lst']
main_list = local_path + '/phd/NextCloud/data/Regions/MetalEarth/larder/j2/main_transect_bb.lst'
out_path = local_path + '/phd/NextCloud//Documents/ME_Transects/Larder/fault_rupture_paper/'
###########################################
# NEW AFTON
# for use_list in ['l0', 'l3', 'l6', 'l9', 'l12', 'l15', 'l18', 'l21']:
# for use_list in ['l3']:
###########################################
# ABITIBI
# list_file = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/j2/upper_abitibi_hex.lst'
# data_file = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/AG/AG_plotset.dat'
###########################################
# TTZ
# out_path = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/snorcle_plots/phi2/'
# # out_path = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/'
# base_path = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/'
# list_file = [base_path + x for x in ('line2a.lst', 'line3.lst', 'line5_2b.lst')]
###########################################
# PLC (Patterson Lake)
# out_path = local_path + '/phd/Nextcloud/data/Regions/plc18/j2/new/plots/'
# base_path = local_path + '/phd/Nextcloud/data/Regions/plc18/j2/new/'
# list_file = [base_path + x for x in ('line1.lst', 'line2.lst')]
###########################################
# WST Anisotropy tests
# out_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/wst/comsol/ellipse_plots/superior_dykes/'
# # out_path = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/'
# base_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/wst/comsol/study15/'
# list_file = [base_path + x for x in ('EW_line5.lst', 'EW_line15.lst', 'EW_line20.lst', 'EW_line25.lst', 'EW_line35.lst',
#                                      'NS_line5.lst', 'NS_line15.lst', 'NS_line20.lst', 'NS_line25.lst', 'NS_line35.lst')]
# list_file = [base_path + x for x in ['line5_2b.lst']]
# list_file = [base_path + x for x in ('line2a.lst', 'line3.lst')]
# main_list = local_path + '/phd/Nextcloud/data/Regions/MetalEarth//j2/LARBB_NS.lst'
# for use_list_dummy in ['LAR','ROU','MAL','MAT']:
# for use_list_dummy in range(0, 40):
# for use_list_dummy in [list_file]:
for use_list_dummy in list_file[:1]:
    #############################
    # Block for WS Comsol models
    main_list = use_list_dummy
    # data = DS.RawData(list_file)
    dataset = DS.Dataset(listfile=main_list)
    # Little hack to get the periods to be the same... Make sure you actually want this on / off
    # dataset.get_data_from_raw(hTol=0.04, lTol=0.12, sites=dataset.raw_data.site_names,
    #                           periods=dataset.raw_data.sites[dataset.raw_data.site_names[0]].periods,
    #                           components=dataset.data.IMPEDANCE_COMPONENTS)
    data = dataset.data
    # data = DS.Data(datafile=data_file, listfile=list_file)
    # data = DS.RawData(main_list)
    # main_transect = DS.Data(datafile=data_file, listfile=list_file)
    # main_transect = DS.RawData(main_list)
    main_transect = deepcopy(dataset.raw_data)
    data.spatial_units = 'km'
    main_transect.spatial_units = 'km'
    # UTM_number = 16
    # UTM_letter = 'N'

    # # data = DS.RawData(listfile=listfile)
    
    # main_transect.remove_sites(sites=[s for s in main_transect.site_names if s not in data.site_names])
    # # Sorting for the WST-Aniso tests
    # if os.path.split(use_list_dummy)[1].startswith('EW'):
    #     x_axis = 'long'
    #     idx = np.argsort(main_transect.locations[:, 0])
    # else:
    #     x_axis = 'lat'
    #     idx = np.argsort(main_transect.locations[:, 1])

    
    # origin = np.flip(main_transect.locations[0, :])
    # For sorting as disctance from a point
    # indices = [0]
    # x = main_transect.locations[:,1]
    # y = main_transect.locations[:,0]
    # for ii in range(main_transect.NS):
    #     dist = [np.linalg.norm(np.array(origin) - np.array((X,Y))) for X,Y in zip(x,y)]
    #     idx = np.argsort(dist)
    #     for ind in idx:
    #         if ind not in indices:
    #             indices.append(ind)
    #             origin = (x[ind], y[ind])
    #             break
    # idx = np.argsort([np.norm(origin - p for p in main_transect.locations)])
    # idx = np.argsort(main_transect.locations[:, 1]) # Sort west-to-east
    idx = np.argsort(main_transect.locations[:, 0]) # Sort south-to-north
    main_transect.site_names = [main_transect.site_names[ii] for ii in idx]
    # main_transect.locations = main_transect.get_locs(mode='latlong')
    x_axis = 'X'
    main_transect.locations = main_transect.locations[idx, :]
    data.site_names = deepcopy(main_transect.site_names)
    data.locations = deepcopy(main_transect.locations)
    dataset.data.site_names = deepcopy(main_transect.site_names)
    dataset.data.locations = deepcopy(main_transect.locations)
    dataset.raw_data.site_names = deepcopy(main_transect.site_names)
    dataset.raw_data.locations = deepcopy(main_transect.locations)
    # for tensor_type in ['phi', 'phi_a', 'Ua', 'Va']:
    for tensor_type in ['phi']:
        # fill_param = ['phi_max', 'phi_min']
        # fill_param = ['phi_split']
        fill_param = ['beta']
        # fill_param = ['azimuth', 'beta']
        # tensor_type = 'phi'
        padding = 20.
        low_cut = 0.001
        high_cut = 600
        save_fig = 1
        freq_skip = 1
        label_offset = -1.2
        xlim = [min(main_transect.locations[:,1])-padding, max(main_transect.locations[:,1])+padding]
        # ylim = [0.5, 4.2]
        ylim = [-2, 4]
        annotate_sites = 0
        backfilled_circle = 0
        separate_colorbar_ticks = 0
        bar_plot = 1
        # out_path = local_path + '/phd/NextCloud/Documents/ME_transects/Upper_Abitibi/Paper/RoughFigures/PT/phi2_beta/'
        # out_path = local_path + '/phd/Nextcloud/data/Regions/synth_test/swayze-esque/PTs/proper_depth/beta/'
        # out_path = local_path + '/phd/Nextcloud/data/Regions/snorcle/j2/jformat-0TN/'
        # out_path = local_path + '/phd/NextCloud/Documents/ME_transects/Rouyn/Edits/'
        # out_file = '{}_phi2-beta_removed_pseudosection'.format(use_list_dummy)
        # out_file = 'line{}_beta_pseudosection'.format(use_list_dummy)
        out_file = '{}_{}_pseudosection'.format(os.path.basename(use_list_dummy).replace('.lst', ''), fill_param[0])

        # file_name = 'MAL_pseudosection_{}_northing'.format(fill_param.replace('_', ''))
        file_exts = ['.png', '.svg']
        dpi = 600


        fig = plt.figure(figsize=(24, 12))
        ax = fig.add_subplot(111)
        MV = gplot.MapView(fig=fig)
        MV.window['figure'] = fig
        MV.window['axes'] = [ax]
        # MV.colourmap = 'turbo'
        MV.colourmap = 'bwr'
        MV.rho_cax = [1, 4]
        MV.diff_cax = [-30, 30]
        # MV.colourmap = 'twilight_shifted'

        MV.site_data['data'] = dataset.data
        MV.site_data['raw_data'] = dataset.raw_data
        MV.site_names = data.site_names
        MV.padding_scale = 10
        MV.pt_scale = 2
        MV.min_pt_ratio = 0.3
        MV.pt_ratio_cutoff = 0.
        MV.phase_error_tol = 1000
        MV.rho_error_tol = 1000
        MV.lut = 64
        MV.ellipse_VE = 1/2 # I had to fudge this and the aspect ratio (further down) to get Longs to plot properly
        MV.ellipse_linewidth = 1
        MV.site_locations['all'] = data.locations
        MV.use_colourbar = True
        # MV.window['figure'] = plt.figure(figsize=(12, 8))
        # MV.window['axes'] = [MV.window['figure'].add_subplot(111)]
        
        if len(fill_param) > 1:
            if backfilled_circle:
                MV.min_pt_ratio = 1
                MV.pt_scale = 2
                # MV.colourmap = 'twilight_shifted'
                MV.ellipse_linewidth = 0
                MV.tensor_ellipse_pseudosection(sites=main_transect.site_names, data_type='raw_data',
                                                fill_param=fill_param[0], periods=(low_cut, high_cut, freq_skip),
                                                pt_type=tensor_type, x_axis=x_axis, annotate_sites=annotate_sites)
                MV.min_pt_ratio = 0.3
                MV.pt_scale = 1.5
                # MV.colourmap = 'twilight_shifted'
                MV.ellipse_linewidth = 0.5
                MV.tensor_ellipse_pseudosection(sites=main_transect.site_names, data_type='raw_data',
                                                fill_param=fill_param[1], periods=(low_cut, high_cut, freq_skip),
                                                pt_type=tensor_type, x_axis=x_axis, annotate_sites=annotate_sites)
            else:
            # MV.colourmap = 'bwr'
                MV.tensor_ellipse_pseudosection(sites=main_transect.site_names, data_type='raw_data',
                                                fill_param=fill_param[0], periods=(low_cut, high_cut, freq_skip),
                                                pt_type=tensor_type, x_axis=x_axis, annotate_sites=annotate_sites)
                MV.tensor_bar_pseudosection(sites=main_transect.site_names, data_type='raw_data',
                                                fill_param=fill_param[1], periods=(low_cut, high_cut, freq_skip),
                                                pt_type=tensor_type, x_axis=x_axis, annotate_sites=annotate_sites)
        else:
            MV.tensor_ellipse_pseudosection(sites=main_transect.site_names, data_type='raw_data',
                                            fill_param=fill_param[0], periods=(low_cut, high_cut, freq_skip),
                                            pt_type=tensor_type, x_axis=x_axis, annotate_sites=annotate_sites)
        linear_site = utils.linear_distance(main_transect.locations[:,1], main_transect.locations[:,0])
        if annotate_sites:
            x = np.array([main_transect.sites[site_name].locations['X'] for site_name in main_transect.site_names])
            y = np.array([main_transect.sites[site_name].locations['Y'] for site_name in main_transect.site_names])
            
            for ii, site in enumerate(main_transect.site_names):
                txt = site[-4:-1]
                if x_axis == 'linear':
                    ax.text(linear_site[ii],
                            label_offset, site, rotation=45)  # 3.6
                else:
                    ax.text(main_transect.sites[site].locations['X'],
                            label_offset, site, rotation=45)  # 3.6
        # MV.set_axis_limits(bounds=[min(linear_site) - 10, max(linear_site) + 10, ylim[0], ylim[1]])
        # MV.set_axis_limits(bounds=[xlim[0], xlim[1], ylim[0], ylim[1]])
        # MV.window['axes'][0].invert_yaxis()
        # if (fill_param[0] != fill_param[1]) and fill_param[1]:
            # two_param = 1
            # MV.plot_phase_bar(data_type='data', normalize=True,
                              # fill_param='beta', period_idx=ii)
        # else:
            # two_param = 0
        # MV.plot_phase_bar2(data_type='data', normalize=True,
        #                    fill_param='phi_min', period_idx=ii)
        # MV.set_axis_limits(bounds=[min(data.locations[:, 1]) - padding,
        #                            max(data.locations[:, 1]) + padding,
        #                            min(data.locations[:, 0]) - padding,
        #                            max(data.locations[:, 0]) + padding])
        # MV.window['axes'][0].set_aspect(1)
        # MV.window['axes'][0].set_xlabel('Distance Along Profile (km)', fontsize=14)
        # MV.window['axes'][0].set_xlabel('Easting (m)', fontsize=14)
        MV.window['axes'][0].set_ylabel('Log10 Period (s)', fontsize=14)
        if len(fill_param) == 1:
            label = MV.get_label(fill_param[0])
            MV.window['colorbar'].set_label(label + r' ($^{\circ}$)',
                                            rotation=270,
                                            labelpad=20,
                                            fontsize=18)
            caxes = [MV.window['colorbar'].ax]
        elif separate_colorbar_ticks:
            cax1 = MV.window['colorbar'].ax
            pos = MV.window['colorbar'].ax.get_position()
            cax1.set_aspect('auto')
            cax2 = MV.window['colorbar'].ax.twinx()
            # MV.window['colorbar'].ax.yaxis.set_label_position('left')
            cax2.set_ylim([-90, 90])
            newlabel = [str(x) for x in range(-90, 100, 10)]
            cax2.set_yticks(range(-90, 100, 10))
            cax2.set_yticklabels(newlabel)
            pos.x0 += 0.05
            pos.x1 += 0.05
            cax1.set_position(pos)
            cax2.set_position(pos)
            label = MV.get_label(fill_param[0])
            cax1.set_ylabel(label + r' ($^{\circ}$)', fontsize=18)
            cax1.yaxis.set_ticks_position('right')
            cax1.yaxis.set_label_position('right')
            cax2.yaxis.set_ticks_position('left')
            cax2.yaxis.set_label_position('left')
            label = MV.get_label(fill_param[1])
            cax2.set_ylabel(label + r' ($^{\circ}$)', fontsize=18)
            caxes = [cax1, cax2]
        # MV.window['axes'][0].set_title('Period: {0:.5g} s'.format(data.periods[ii]))
        # ells, vals, norm_vals = plot_ellipse(data, fill_param='phi_max')
        if save_fig:
            for file_format in file_exts:
                plt.savefig(out_path + out_file + file_format, dpi=dpi,
                            transparent=True)
            plt.close('all')
            MV.window['colorbar'] = None
            # ax.clear()

            # for x in caxes:
            #     x.clear()
        else:
            plt.show()
