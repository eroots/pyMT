import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pyMT.utils as utils
import os


data_UD = DS.Data(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/LARall.lst',
	              datafile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/Larder_HexMT_all_resp.data')
C = np.load('E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/f-dependent_distortions.npy')
plot_data = False
write_data = True
path = 'E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/resps/shifted/'
files = os.listdir('E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/resps/')
files = [x for x in files if 'resp' in x]
for file_out in files:
	data_R = DS.Data(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/LARall.lst',
					 datafile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/resps/{}'.format(file_out))
	# data_R = DS.Data(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/LARall.lst',
	# 				 datafile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/Hex2Mod-LL_base_resp.dat')
	data_D = deepcopy(data_R)

	for ii, site in enumerate(data_R.site_names):
		zxxr = data_R.sites[site].data['ZXXR']-1j*data_R.sites[site].data['ZXXI']
		zxyr = data_R.sites[site].data['ZXYR']-1j*data_R.sites[site].data['ZXYI']
		zyxr = data_R.sites[site].data['ZYXR']-1j*data_R.sites[site].data['ZYXI']
		zyyr = data_R.sites[site].data['ZYYR']-1j*data_R.sites[site].data['ZYYI']
		Z_R = np.array([[zxxr, zxyr], [zyxr, zyyr]])

		for ip in range(data_R.sites[site].NP):
			Z_D = np.matmul(np.reshape(C[:,ii,ip], [2, 2]), Z_R[:,:,ip])

			data_D.sites[site].data['ZXXR'][ip] = np.real(Z_D[0,0])
			data_D.sites[site].data['ZXYR'][ip] = np.real(Z_D[0,1])
			data_D.sites[site].data['ZYXR'][ip] = np.real(Z_D[1,0])
			data_D.sites[site].data['ZYYR'][ip] = np.real(Z_D[1,1])

			data_D.sites[site].data['ZXXI'][ip] = -1*np.imag(Z_D[0,0])
			data_D.sites[site].data['ZXYI'][ip] = -1*np.imag(Z_D[0,1])
			data_D.sites[site].data['ZYXI'][ip] = -1*np.imag(Z_D[1,0])
			data_D.sites[site].data['ZYYI'][ip] = -1*np.imag(Z_D[1,1])

	if write_data:
		data_D.write(path + file_out)

if plot_data:
	rhoxy_d = utils.compute_rho(data_D.sites[site], calc_comp='xy')[0]
	phaxy_d = utils.compute_phase(data_D.sites[site], calc_comp='xy', wrap=1)[0]
	rhoyx_d = utils.compute_rho(data_D.sites[site], calc_comp='yx')[0]
	phayx_d = utils.compute_phase(data_D.sites[site], calc_comp='yx', wrap=1)[0]

	rhoxy_r = utils.compute_rho(data_R.sites[site], calc_comp='xy')[0]
	phaxy_r = utils.compute_phase(data_R.sites[site], calc_comp='xy', wrap=1)[0]
	rhoyx_r = utils.compute_rho(data_R.sites[site], calc_comp='yx')[0]
	phayx_r = utils.compute_phase(data_R.sites[site], calc_comp='yx', wrap=1)[0]

	rhoxy_ud = utils.compute_rho(data_UD.sites[site], calc_comp='xy')[0]
	phaxy_ud = utils.compute_phase(data_UD.sites[site], calc_comp='xy', wrap=1)[0]
	rhoyx_ud = utils.compute_rho(data_UD.sites[site], calc_comp='yx')[0]
	phayx_ud = utils.compute_phase(data_UD.sites[site], calc_comp='yx', wrap=1)[0]

	plt.subplot(2,1,1)
	plt.plot(np.log10(data_UD.periods), np.log10(rhoxy_d), 'b.')
	plt.plot(np.log10(data_UD.periods), np.log10(rhoxy_r), 'bx')
	plt.plot(np.log10(data_UD.periods), np.log10(rhoxy_ud), 'b--')
	plt.plot(np.log10(data_UD.periods), np.log10(rhoyx_d), 'r.')
	plt.plot(np.log10(data_UD.periods), np.log10(rhoyx_r), 'rx')
	plt.plot(np.log10(data_UD.periods), np.log10(rhoyx_ud), 'r--')

	plt.subplot(2,1,2)
	plt.plot(np.log10(data_UD.periods), (phaxy_d), 'b.')
	plt.plot(np.log10(data_UD.periods), (phaxy_r), 'bx')
	plt.plot(np.log10(data_UD.periods), (phaxy_ud), 'b--')
	plt.plot(np.log10(data_UD.periods), (phayx_d), 'r.')
	plt.plot(np.log10(data_UD.periods), (phayx_r), 'rx')
	plt.plot(np.log10(data_UD.periods), (phayx_ud), 'r--')

	plt.show()
