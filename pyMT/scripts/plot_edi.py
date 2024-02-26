import pyMT.utils as utils
import pyMT.data_structures as DS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import sys
# from pyMT import gplot


plot_options = {'marker_dict': {'Rxy':  'o',
								'Rxx':  'o',
								'Ryy':  'o',
								'Ryx':  'o',
								'Pxy':  'o',
								'Pxx':  'o',
								'Pyy':  'o',
								'Pyx':  'o',
								'TZXR': 'o',
								'TZXI': 'o',
								'TZYR': 'o',
								'TZYI': 'o'},
				'linestyle': '',
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
				'alpha': 	   {'Rxy':  1,
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
				'markersize': 6,
				'edgewidth': 1,
				'x_limits': [1e-3, 1e4],
				'y_limits':     {'R': [1e0, 1e5],
								 'P': [0, 120],
								 'T': [-0.75, 0.75]}
								 }

def plot_site(site):
	to_plot = []
	if 'ZXYR' in site.components:
		to_plot.append('Rxy')	
		to_plot.append('Ryx')
		to_plot.append('Rxx')
		to_plot.append('Ryy')
		# to_plot.append('PXX')
		to_plot.append('Pxy')
		to_plot.append('Pyx')
		# to_plot.append('PYY')
	if 'TZXR' in site.components:
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
	fig = plt.figure()
	bottom_plot = 'tzi'
	top_plot = 'rho'
	if len(to_plot) == 6:
		spec['rho'] = GridSpec(3, 3, figure=fig).new_subplotspec((0, 0), colspan=3, rowspan=2)
		spec['phase'] = GridSpec(3, 3, figure=fig).new_subplotspec((2, 0), colspan=3, rowspan=1)
		axes['rho'] = fig.add_subplot(spec['rho'])
		axes['phase'] = fig.add_subplot(spec['phase'])
		bottom_plot = 'phase'
	elif len(to_plot) == 4:
		top_plot = 'tzr'
		spec['tzr'] = GridSpec(2, 2, figure=fig).new_subplotspec((0, 0), colspan=2, rowspan=1)
		spec['tzi'] = GridSpec(2, 2, figure=fig).new_subplotspec((1, 0), colspan=2, rowspan=1)
		spec['tzr'] = fig.add_subplot(spec['tzr'])
		spec['tzi'] = fig.add_subplot(spec['tzi'])
	elif len(to_plot) > 6:
		spec['rho'] = GridSpec(6, 6, figure=fig).new_subplotspec((0, 0), colspan=6, rowspan=3)
		spec['phase'] = GridSpec(6, 6, figure=fig).new_subplotspec((3, 0), colspan=6, rowspan=1)
		spec['tzr'] = GridSpec(6, 6, figure=fig).new_subplotspec((4, 0), colspan=6, rowspan=1)
		spec['tzi'] = GridSpec(6, 6, figure=fig).new_subplotspec((5, 0), colspan=6, rowspan=1)
		axes['rho'] = fig.add_subplot(spec['rho'])
		axes['phase'] = fig.add_subplot(spec['phase'])
		axes['tzr'] = fig.add_subplot(spec['tzr'])
		axes['tzi'] = fig.add_subplot(spec['tzi'])

	for tp in to_plot:
		if tp.startswith('R'):
			rho, e, log_e = utils.compute_rho(site, calc_comp=tp[1:], errtype='errors')
			axes['rho'].errorbar((site.periods), (rho), xerr=None, yerr=e,
						         marker=plot_options['marker_dict'][tp],
						         linestyle=plot_options['linestyle'],
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
			pha, e = utils.compute_phase(site, calc_comp=tp[1:], errtype='errors', wrap=True)
			axes['phase'].errorbar((site.periods), pha, xerr=None, yerr=e,
						           marker=plot_options['marker_dict'][tp],
						           linestyle=plot_options['linestyle'],
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
			comp = 'tz{}'.format(tp[-1].lower())
			axes[comp].errorbar((site.periods), 
								 site.data[tp],
								 xerr=None, yerr=site.errors[tp],
						         marker=plot_options['marker_dict'][tp],
						         linestyle=plot_options['linestyle'],
						         color=plot_options['colour_dict'][tp],
						         mec=plot_options['edgecolor'],
						         markersize=plot_options['markersize'],
						         mew=plot_options['edgewidth'],
						         alpha=plot_options['alpha'][tp],
						         label=tp)
			axes[comp].set_xscale('log')
			if comp.endswith('r'):
				axes[comp].set_ylabel(r'Re {Tz}')
			else:
				axes[comp].set_ylabel(r'Im {Tz}')
			axes[comp].set_ylim(plot_options['y_limits']['T'])
			axes[comp].set_xlim(plot_options['x_limits'])
	for ax in axes.values():
		if ax:
			ax.legend(loc='upper left')
	axes[bottom_plot].set_xlabel('Period (s)')
	axes[top_plot].set_title(site.name)
	plt.show()


if __name__ == '__main__':
	# file = 'E:/Work/Regions/ATHA/Atha21_emtf/zxx/10124_2021-09-15-222830/13_flipHy_SL1.edi'
	data = DS.RawData(sys.argv[1])
	# data = DS.RawData(file)
	plot_site(data.sites[data.site_names[0]])
