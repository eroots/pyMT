import pyMT.data_structures as DS
import matplotlib.pyplot as plt
import pyMT.gplot as gplot


def generate_misfit_plots(dataset, save_path, site_names=None, figure_size=(8, 6), file_format='.png'):
    fig = plt.figure(figsize=figure_size)
    dpm = gplot.DataPlotManager(fig=fig)
    if not site_names:
        site_names = dataset.data.site_names
    components = {'impedance_diagonal': ('ZXXR', 'ZXXI', 'ZYYR', 'ZYYI'),
                  'impedance_off-diagonal': ('ZXYR', 'ZXYI', 'ZYXR', 'ZYXI'),
                  'tipper': ('TZXR', 'TZXI', 'TZYR', 'TZYI')
                  # 'tipper_real': ('TZXR', 'TZYR'),
                  # 'tipper_imag': ('TZXI', 'TZYI'),
                  # 'phase_diagonal': ('PhaXX', 'PhaYY'),
                  # 'rho_diagonal': ('RhoXX', 'RhoYY'),
                  # 'rho_off-diagonal': ('RhoXY', 'RhoYX'),
                  # 'phase_off-diagonal': ('PhaXY', 'PhaYX')}
                  }
    dpm.toggles = {'raw_data': False,
                   'data': True,
                   'response': True}
    dpm.ax_lim_dict = {'rho': [0, 5], 'phase': [0, 120], 'impedance': [-0.3, 0.3],
                    'tipper': [-0.6, 0.6], 'beta': [-10, 10], 'azimuth': [-90, 90]}
    dpm.markersize = 5 # Scale according to figure size
    dpm.edgewidth  = 0.1
    dpm.link_axes_bounds = True
    dpm.label_fontsize = 10 # Scale according to figure size
    dpm.outlier_thresh = 5
    dpm.wrap = True
    dpm.plot_flagged_data = False
    dpm.show_outliers = False
    dpm.sites = dataset.get_sites(site_names=site_names[0], dTypes=['data', 'response'])
    dpm.sites.update({'1d': []})
    dpm.sites.update({'smoothed_data': []})
    dpm.redraw_axes()

    if 'TZXR' not in dataset.data.components:
        del components['tipper_real']
        del components['tipper_imag']
        del components['tipper']
    for comps in components.keys():
        # if 'phase' in comps or 'rho' in comps:
        #     dpm.link_axes_bounds = True
        # else:
        #     dpm.link_axes_bounds = False
        print('Generating {} plots...'.format(comps))
        dpm.components = components[comps]
        for site in site_names:
            if 'tipper' in comps:
                dpm.scale = 'none'
            else:
                dpm.scale = 'sqrt(periods)'
            if comps == 'impedance_diagonal':
                dpm.ax_lim_dict['impedance'] = [-0.15, 0.15]
            elif comps == 'impedance_off-diagonal':
                dpm.ax_lim_dict['impedance'] = [-0.3, 0.3]
            dpm.sites = dataset.get_sites(site_names=site, dTypes=['data', 'response'])
            dpm.sites.update({'1d': []})
            dpm.sites.update({'smoothed_data': []})
            dpm.which_errors = ['data']
            # dpm.plot_data()
            dpm.redraw_single_axis(site_name=site)
            dpm.fig.savefig(save_path + site + '_' + comps + file_format,
                            format=file_format.strip('.'),
                            dpi=600,
                            bbox_inches='tight')


dataset = DS.Dataset(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst',
                     datafile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wst_cullmantle3_LAMBERT_ZK_removed.dat',
                     responsefile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/anisotropic/wstZK_ani_lastIter.dat')

generate_misfit_plots(dataset,
                      save_path='E:/phd/NextCloud/Documents/ME_Transects/wst-mantle/rms_plots/fit_examples/svgs/',
                      site_names=['WST10', 'WST36', 'WST21', 'WST98', 'WST57', 'site214', 'WST128', 'WST155'],
                      figure_size=(6, 3),
                      file_format='.svg')