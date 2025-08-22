import pyMT.utils as utils
from matplotlib.backends.backend_pdf import PdfPages
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
import os
# from pyMT import gplot


plot_options = {'marker_dict': {'data': {'Rxy':  'o',
                                         'Rxx':  'v',
                                         'Ryy':  'v',
                                         'Ryx':  'o',
                                         'Pxy':  'o',
                                         'Pxx':  'v',
                                         'Pyy':  'v',
                                         'Pyx':  'o',
                                         'TZXR': 'o',
                                         'TZXI': 'o',
                                         'TZYR': 'o',
                                         'TZYI': 'o'},
                                 'resp': {'Rxy':  '',
                                          'Rxx':  '',
                                          'Ryy':  '',
                                          'Ryx':  '',
                                          'Pxy':  '',
                                          'Pxx':  '',
                                          'Pyy':  '',
                                          'Pyx':  '',
                                          'TZXR': '',
                                          'TZXI': '',
                                          'TZYR': '',
                                          'TZYI': ''}},

                'linestyle': {'data': {'Rxy':  '',
                                       'Rxx':  '',
                                       'Ryy':  '',
                                       'Ryx':  '',
                                       'Pxy':  '',
                                       'Pxx':  '',
                                       'Pyy':  '',
                                       'Pyx':  '',
                                       'TZXR': '',
                                       'TZXI': '',
                                       'TZYR': '',
                                       'TZYI': ''},
                              'resp': {'Rxy':  '-',
                                       'Rxx':  '--',
                                       'Ryy':  '-',
                                       'Ryx':  '--',
                                       'Pxy':  '-',
                                       'Pxx':  '--',
                                       'Pyy':  '--',
                                       'Pyx':  '-',
                                       'TZXR': '-',
                                       'TZXI': '--',
                                       'TZYR': '-',
                                       'TZYI': '--'}},
                'colour_dict': {'Rxy':  'b',
                                'Rxx':  'b',
                                'Ryy':  'r',
                                'Ryx':  'r',
                                'Pxy':  'b',
                                'Pxx':  'b',
                                'Pyy':  'r',
                                'Pyx':  'r',
                                'TZXR': 'b',
                                'TZXI': 'b',
                                'TZYR': 'r',
                                'TZYI': 'r'},
                'alpha':       {'Rxy':  1,
                                'Rxx':  0.5,
                                'Ryy':  0.5,
                                'Ryx':  1,
                                'Pxy':  1,
                                'Pxx':  0.5,
                                'Pyy':  0.5,
                                'Pyx':  1,
                                'TZXR': 1,
                                'TZXI': 1,
                                'TZYR': 1,
                                'TZYI': 1},
                'edgecolor': 'k',
                'markersize': 4,
                'edgewidth': 1,
                'x_limits': [1e-3, 1e4],
                'y_limits':     {'R': [1e0, 1e5],
                                 'P': [-180, 180],
                                 'T': [-0.75, 0.75]}
                                 }

def plot_sites(sites, fig, resp_sites=None, data_type=['data'], save_fig=False, plots_per_row=2):
    site_names = [site.name for site in sites['data']] # Assume at minimum the data has been passed in
    for ss, site_name in enumerate(site_names):
        to_plot = []
        # Assume 6 sites per page
        # Always plot tipper, just make it empty if its not there
        # if 'ZXYR' in site.components:
        #     to_plot.append('Rxy')   
        #     to_plot.append('Ryx')
        #     to_plot.append('Rxx')
        #     to_plot.append('Ryy')
        #     # to_plot.append('PXX')
        #     to_plot.append('Pxy')
        #     to_plot.append('Pyx')
        #     # to_plot.append('PYY')
        # if 'TZXR' in site.components:
        #     to_plot.append('TZXR')
        #     to_plot.append('TZYR')
        #     to_plot.append('TZXI')
        #     to_plot.append('TZYI')
        # if 'ZXYR' in site.components:
        to_plot.append('Rxy')   
        to_plot.append('Ryx')
        to_plot.append('Rxx')
        to_plot.append('Ryy')
        to_plot.append('Pxx')
        to_plot.append('Pxy')
        to_plot.append('Pyx')
        to_plot.append('Pyy')
        # if 'TZXR' in site.components:
        to_plot.append('TZXR')
        to_plot.append('TZYR')
        to_plot.append('TZXI')
        to_plot.append('TZYI')

        axes = {'rho': None,
                'phase': None,
                'tzr': None,
                'tzy': None}
        spec = {'rho': None,
                'phase': None,
                'tzr': None,
                'tzi': None}
        # fig = plt.figure(figsize=(10, 12))
        bottom_plot = 'tz'
        top_plot = 'rho'
        # if len(to_plot) == 6:
        #     spec['rho'] = GridSpec(3, 3, figure=fig).new_subplotspec((0, 0), colspan=3, rowspan=2)
        #     spec['phase'] = GridSpec(3, 3, figure=fig).new_subplotspec((2, 0), colspan=3, rowspan=1)
        #     axes['rho'] = fig.add_subplot(spec['rho'])
        #     axes['phase'] = fig.add_subplot(spec['phase'])
        #     bottom_plot = 'phase'
        # elif len(to_plot) == 4:
        #     top_plot = 'tzr'
        #     spec['tzr'] = GridSpec(2, 2, figure=fig).new_subplotspec((0, 0), colspan=2, rowspan=1)
        #     spec['tzi'] = GridSpec(2, 2, figure=fig).new_subplotspec((1, 0), colspan=2, rowspan=1)
        #     spec['tzr'] = fig.add_subplot(spec['tzr'])
        #     spec['tzi'] = fig.add_subplot(spec['tzi'])
        # elif len(to_plot) > 6:
        if ss < 2:
            row_number = 0
            col_number = ss*8 + ss
        else:
            row_number = 9
            col_number = (ss % plots_per_row) * 8 + (ss % plots_per_row)
        spec['rho'] = GridSpec(17, 17, figure=fig).new_subplotspec((row_number, col_number), colspan=8, rowspan=4)
        spec['phase'] = GridSpec(17, 17, figure=fig).new_subplotspec((row_number+4, col_number), colspan=8, rowspan=2)
        spec['tz'] = GridSpec(17, 17, figure=fig).new_subplotspec((row_number+6, col_number), colspan=8, rowspan=2)
        # spec['tzi'] = GridSpec(6, 6, figure=fig).new_subplotspec((5, 0), colspan=2, rowspan=1)
        axes['rho'] = fig.add_subplot(spec['rho'])
        axes['phase'] = fig.add_subplot(spec['phase'])
        axes['tz'] = fig.add_subplot(spec['tz'])
        # axes['tzi'] = fig.add_subplot(spec['tzi'])
        for dtype in data_type:
            site = sites[dtype][ss]
            for tp in to_plot:
                if tp.startswith('R'):
                    rho, e, log_e = utils.compute_rho(site, calc_comp=tp[1:], errtype='errors')
                    axes['rho'].errorbar((site.periods), (rho), xerr=None, yerr=e,
                                         marker=plot_options['marker_dict'][dtype][tp],
                                         linestyle=plot_options['linestyle'][dtype][tp],
                                         color=plot_options['colour_dict'][tp],
                                         mec=plot_options['edgecolor'],
                                         markersize=plot_options['markersize'],
                                         mew=plot_options['edgewidth'],
                                         alpha=plot_options['alpha'][tp],
                                         label=tp)
                    axes['rho'].set_ylabel('App. Rho (ohm-m)')
                    axes['rho'].set_ylim(plot_options['y_limits']['R'])
                    axes['rho'].set_xlim(plot_options['x_limits'])
                    axes['rho'].set_xscale('log')
                    axes['rho'].set_yscale('log')
                elif tp.startswith('P'):
                    pha, e = utils.compute_phase(site, calc_comp=tp[1:], errtype='errors', wrap=False)
                    axes['phase'].errorbar((site.periods), pha, xerr=None, yerr=e,
                                           marker=plot_options['marker_dict'][dtype][tp],
                                           linestyle=plot_options['linestyle'][dtype][tp],
                                           color=plot_options['colour_dict'][tp],
                                           mec=plot_options['edgecolor'],
                                           markersize=plot_options['markersize'],
                                           mew=plot_options['edgewidth'],
                                           alpha=plot_options['alpha'][tp],
                                           label=tp)
                    axes['phase'].set_ylabel('Phase (Degrees)')
                    axes['phase'].set_ylim(plot_options['y_limits']['P'])
                    axes['phase'].set_xlim(plot_options['x_limits'])
                    axes['phase'].set_xscale('log')
                elif tp.startswith('T'):
                    # comp = 'tz{}'.format(tp[-1].lower())
                    comp = 'tz'
                    if tp in site.components:
                        axes[comp].errorbar((site.periods), 
                                             site.data[tp],
                                             xerr=None, yerr=site.errors[tp],
                                             marker=plot_options['marker_dict'][dtype][tp],
                                             linestyle=plot_options['linestyle'][dtype][tp],
                                             color=plot_options['colour_dict'][tp],
                                             mec=plot_options['edgecolor'],
                                             markersize=plot_options['markersize'],
                                             mew=plot_options['edgewidth'],
                                             alpha=plot_options['alpha'][tp],
                                             label=tp)
                    axes[comp].set_xscale('log')
                    axes[comp].grid(True)
                    axes[comp].set_ylabel('Tz')
                    # if comp.endswith('r'):
                    #     axes[comp].set_ylabel(r'Re {Tz}')
                    # else:
                    #     axes[comp].set_ylabel(r'Im {Tz}')
                    axes[comp].set_ylim(plot_options['y_limits']['T'])
                    axes[comp].set_xlim(plot_options['x_limits'])
            for ax_name, ax in axes.items():
                if ax:
                    if ss == 0:
                        ax.legend(loc='upper left')
                    if ax_name != bottom_plot:
                        ax.set_xticklabels([])
                    if col_number > 0:
                        ax.set_yticklabels([])
                        ax.set_ylabel('')
        axes[bottom_plot].set_xlabel('Period (s)')
        axes[top_plot].set_title(site.name)
    if save_fig:
        plt.savefig(site.name)
        plt.close()
    else:
        return fig

# def main():
#     if sys.argv[1].lower() == 'all':
#         for site in os.listdir():
#             if site.endswith('.edi'):
#                 data = DS.RawData(site)
#                 plot_site(data.sites[data.site_names[0]], save_fig=True)
#     else:
#         data = DS.RawData(sys.argv[1])
#         plot_site(data.sites[data.site_names[0]], save_fig=False)


# if __name__ == '__main__':
#     main()
    # file = 'E:/Work/Regions/ATHA/Atha21_emtf/zxx/10124_2021-09-15-222830/13_flipHy_SL1.edi'
    
    # data = DS.RawData(file)
    




listfile = r'E:/Work/sync/Regions/ATHA/j2/all_wTHOT.lst'
datafile = r'E:/Work/sync/Regions/ATHA/gofem/model6/atha5_gofem-ZK_rm-sites.gdat'
respfile = r'E:/Work/sync/Regions/ATHA/gofem/model6/ZK2/inv_data_42.gdat'
# listfile = r'E:/Work/sync/Regions/ATHA/j2/tie_line.lst'
# dataset = DS.Dataset(listfile=listfile)
dataset = DS.Dataset(datafile=datafile, responsefile=respfile)
out_path = 'E:/work/sync/Documents/ATHA/NRCAN_0001/'


sites_per_page = 4
with PdfPages(''.join([out_path, 'atha_bb_misfit-plots.pdf'])) as pdf:
    for ii in range(0, dataset.data.NS, sites_per_page):
        fig = plt.figure(figsize=(12, 16))
        site_names = dataset.data.site_names[ii:ii + sites_per_page]
        # data_sites = [dataset.raw_data.sites[site] for site in site_names]
        data_sites = [dataset.data.sites[site] for site in site_names]
        resp_sites = [dataset.response.sites[site] for site in site_names]
        sites = {'data': data_sites, 'resp': resp_sites}
        plot_sites(sites, fig=fig, data_type=['data', 'resp'], save_fig=False, plots_per_row=2)
        pdf.savefig()
        # plt.close()
