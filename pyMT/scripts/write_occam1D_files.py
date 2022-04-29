import pyMT.data_structures as DS
import numpy as np
import pyMT.utils as utils
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt


# data_file = 'E:/phd/NextCloud/data/Regions/Ciomadul/cio_allpers.dat'
# model_file = 'E:/phd/NextCloud/data/Regions/Ciomadul/cio5/1D/smoothed/diffmesh/cio1D-2.model'
# output_path = 'E:/phd/NextCloud/data/Regions/Ciomadul/1D_inversions/'
# data_file = 'E:/phd/NextCloud/data/Regions/TTZ/full_run/ttz_full_cull1.dat'
# model_file = 'E:/phd/NextCloud/data/Regions/TTZ/full_run/ttz_hs2500.rho'
# output_path = 'E:/phd/NextCloud/data/Regions/TTZ/1D_inversions/test_imp/'
data_file = 'E:/phd/NextCloud/data/Regions/snorcle/j2/jformat-0TN/j2edi/ffmt_output/renamed/sorted_cull1.lst'
# model_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/full_run/ttz_hs2500.rho'
runtime_path = 'C:/Users/eroots/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/allsites_ssq/'
output_path = 'E:/phd/NextCloud/data/Regions/snorcle/1D_inversions/normal/allsites_ssq/'
data = DS.RawData(data_file)
# ds = DS.Dataset(listfile=data_file)
# ds.get_data_from_raw(periods=list(ds.raw_data.narrow_periods.keys()))
# data = ds.data
low_cut = 0.0001
high_cut = 300
remove_periods = True
use_component = 'ssq'
use_average = False
variable_rho = 1
fixed_rho = 2500
perc_err = 0.5
phase_error = 2.5
output_name = 'startup'
# occam_model_name = 'SWZ1D_model.txt'
# occam_data_name = 'SWZ1D_data_pha_'
datestr = date.today()
datestr = datestr.strftime('%d/%m/%y')
max_iter = 20
target_misfit = 0.5
roughness_type = 1
model_limits = [-2, 6]
model_value_steps = [0]
iteration_num = 0
lagrange = 5
use_phase = 1
use_rho = 1
use_imp = 0
# use_dz = model.dz
use_dz = [0] + list(np.logspace(0, 6, 300))

if remove_periods:
	for site in data.site_names:
		s = data.sites[site]
		remove_periods = s.periods[(s.periods < low_cut) + (s.periods > high_cut)]
		data.sites[site].remove_periods(periods=remove_periods)
# model = DS.Model(model_file)
rho = {site: utils.compute_rho(data.sites[site], calc_comp=use_component)[0] for site in data.site_names}
phase = {site: utils.compute_phase(data.sites[site], calc_comp=use_component)[0] for site in data.site_names}
zr = {site: (data.sites[site].data['ZXYR'] - data.sites[site].data['ZYXR']) / 2 for site in data.site_names}
zi = {site: (-data.sites[site].data['ZXYI'] + data.sites[site].data['ZYXI']) / 2 for site in data.site_names}
##############
# If using a data file
# rho_smooth = {site: utils.geotools_filter(data.periods, rho[site], fwidth=0.65) for site in data.site_names}
# phase_smooth = {site: utils.geotools_filter(data.periods, phase[site], fwidth=0.65) for site in data.site_names}
# zr_smooth = {site: utils.geotools_filter(data.periods, zr[site], fwidth=0.65) for site in data.site_names}
# zi_smooth = {site: utils.geotools_filter(data.periods, zi[site], fwidth=0.65) for site in data.site_names}
##############
# site = data.site_names[0]

param_count = len(use_dz)
# model.nz
# If using a list file
rho_smooth = {site: utils.geotools_filter(data.sites[site].periods, rho[site], fwidth=0.65) for site in data.site_names}
phase_smooth = {site: utils.geotools_filter(data.sites[site].periods, phase[site], fwidth=0.65) for site in data.site_names}
zr_smooth = {site: utils.geotools_filter(data.sites[site].periods, zr[site], fwidth=0.65) for site in data.site_names}
zi_smooth = {site: utils.geotools_filter(data.sites[site].periods, zi[site], fwidth=0.65) for site in data.site_names}

if use_average:
	average_smooth_rho = 10 ** np.mean([np.log10(rho_smooth[site]) for site in data.site_names], axis=0)
	average_smooth_phase = np.mean([phase_smooth[site] for site in data.site_names], axis=0)
	site_names = ['average_site']
	rho_smooth = {'average_site': average_smooth_rho}
	phase_smooth = {'average_site': average_smooth_phase}
	# rho_smooth = {'average_site': average_smooth_rho}
else:
	site_names = data.site_names
for site in site_names:
	NP = len(rho_smooth[site])
	NS = len(rho_smooth)
	try:
		periods = data.periods
	except AttributeError:
		periods = data.sites[site].periods
	## Uncomment if using raw data
	# periods = data.sites[site].periods
	if use_phase and use_rho:
		num_data = NP * 2
	else:
		num_data = NP	
	if variable_rho:
		starting_rho = 10 ** np.mean(np.log10(rho_smooth[site]))
	else:
		starting_rho = fixed_rho
# for site in ['CS_site5']:
	full_runtime_path = runtime_path + site + '/'
	full_output_path = output_path + site + '/'
	Path(full_output_path).mkdir(parents=True, exist_ok=True)
	output_name = full_output_path + 'startup'
	occam_model_name = 'SWZ1D_model.txt'
	occam_data_name = 'SWZ1D_data_' + site + '.txt'
	# starting_rho = np.mean(rho_smooth[site])
# STARTUP FILE
	header = '\n'.join(['Format: OCCAMITER_FLEX ! Flexible format',
						'Description: test ! Not used by Occam, but you can use this for your own notes.',
						'Model File: {} ! Name of the Model File. Case sensitive.',
						'Data File: {} ! Name of the Data File. Case sensitive.',
						'Date/Time: {} ! On output, date and time stamp placed here.',
						'Max Iter: {} ! Maximum number of Occam iterations to compute.',
						'Target Misfit: {} ! Target RMS misfit, see equation 2.',
						'Roughness Type: {} ! See section below for Roughness Penalty options',
						 # 'Model Limits: {},{} ! Optional, places hard limits on log10(rho) values.',
						'!Model Limits:  ! Optional, places hard limits on log10(rho) values.',
						'!Model Value Steps: stepsize ! Optional, forces model into discrete steps of stepsize.',
						'Debug Level: 1 ! Console output. 0: minimal, 1: default, 2: detailed',
						'Iteration: 0 ! Iteration number, use 0 for starting from scratch.',
						'Lagrange Value: {} ! log10(largrance multiplier), starting value.',
						'Roughness Value: 0.1000000E+08 ! Roughness of last model, ignored on startup',
						'Misfit Value: 100.0000 ! Misfit of model listed below. Ignored on startup',
						'Misfit Reached: 0 ! 0: not reached, 1: reached. Useful when restarting.',
						'Param Count: {} ! Number of free inversion parameters.\n']).format(occam_model_name,
																						  occam_data_name,
																						  datestr,
																						  max_iter,
																						  target_misfit,
																						  roughness_type,
																						  # model_limits[0],
																						  # model_limits[1],
																						  lagrange,
																						  param_count)

	with open(output_name, 'w') as f:
		f.write(header)
		for ii in range(param_count):
			f.write('{:>6.3g}\n'.format(np.log10(starting_rho)))


	# MODEL FILE
	header = '\n'.join(['Format: Resistivity1DMod_1.0',
						'#Layers: {:>10d}'.format(len(use_dz) + 1),
						'! Layer block listing is:\n'
						'! [top_depth resistivity penalty preference pref_penalty]',
						'{:>13.2f}{:>13s}{:>13d}{:>13d}{:>13d}\n'.format(-100000, '1d12', 0, 0, 0)])
	with open(full_output_path + occam_model_name, 'w') as f:
		f.write(header)
		for ii in range(param_count):
			f.write('{:>13.2f}{:>13d}{:>13d}{:>13d}{:>13d}\n'.format(use_dz[ii], -1, 1, 0, 0))

# DATA FILE

	
	with open(full_output_path + occam_data_name, 'w') as f:
		header = '\n'.join(['Format: EMData_1.2',
							'Dipole Length: 0 ! [m]',
							'# integ pts: 0 ! Optional and only used if Dipole Length > 0.',
							'# Transmitters: 1\n'])
		trans_block = '\n'.join(['!{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}'.format('X', 'Y', 'Z', 'AZIMUTH', 'DIP'),
								 '{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n'.format('0', '0', '0', '90', '0')])
		freq_block_header = '# Frequencies: {:>13d}\n'.format(NP)
		f.write(header)
		f.write(trans_block)
		f.write(freq_block_header)
		for p in periods:
			f.write('{:>13.5g}\n'.format(1 / p))
		f.write('# Receivers:     1\n')	
		f.write('!{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n'.format('X', 'Y', 'Z', 'THETA', 'ALPHA', 'BETA'))
		f.write('{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n'.format('0', '0', '0', '0', '0', '0'))
		f.write('# Data:     {}\n'.format(num_data))
		f.write('!{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}{:>13s}\n'.format('TYPE', 'FREQ#', 'TX#', 'RX#', 'DATA', 'SD_ERROR'))
		for ii in range(NP):
			if use_rho:
				f.write('{:>13s}{:>13d}{:>13d}{:>13d}{:>13.5e}{:>13.5e}\n'.format('RhoZxy', ii + 1, 1, 1, rho_smooth[site][ii], rho_smooth[site][ii] * perc_err))
			if use_phase:
				f.write('{:>13s}{:>13d}{:>13d}{:>13d}{:>13.5e}{:>13.5e}\n'.format('PhsZyx', ii + 1, 1, 1, phase_smooth[site][ii], phase_error))
			if use_imp:
				f.write('{:>13s}{:>13d}{:>13d}{:>13d}{:>13.5e}{:>13.5e}\n'.format('RealZxy', ii + 1, 1, 1, zr[site][ii], zr[site][ii] * perc_err))
				f.write('{:>13s}{:>13d}{:>13d}{:>13d}{:>13.5e}{:>13.5e}\n'.format('ImagZxy', ii + 1, 1, 1, zi[site][ii], zi[site][ii] * perc_err))

	# plt.loglog(data.periods, phase[site], 'ko')
	# plt.loglog(data.periods, phase_smooth[site], 'r--')
	# plt.gcf().savefig(full_output_path + 'rhoplot.png')
	# plt.close()