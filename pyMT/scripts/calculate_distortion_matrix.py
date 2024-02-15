import pyMT.data_structures as DS
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pyMT.utils as utils


i_site = 10
data_D = DS.Data(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/LARall.lst',
	             datafile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/Larder_HexMT_all_resp.data')
data_R = DS.Data(listfile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/j2/LARall.lst',
				 datafile='E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/Hex2Mod-LL_base_resp.dat')

data_UD = deepcopy(data_D)

all_C = np.zeros((4, data_UD.NS, data_UD.NP))
for ii, site in enumerate(data_R.site_names):
	zxxr = data_R.sites[site].data['ZXXR']-1j*data_R.sites[site].data['ZXXI']
	zxyr = data_R.sites[site].data['ZXYR']-1j*data_R.sites[site].data['ZXYI']
	zyxr = data_R.sites[site].data['ZYXR']-1j*data_R.sites[site].data['ZYXI']
	zyyr = data_R.sites[site].data['ZYYR']-1j*data_R.sites[site].data['ZYYI']
	Z_R = np.array([[zxxr, zxyr], [zyxr, zyyr]])

	zxxd = data_D.sites[site].data['ZXXR']-1j*data_D.sites[site].data['ZXXI']
	zxyd = data_D.sites[site].data['ZXYR']-1j*data_D.sites[site].data['ZXYI']
	zyxd = data_D.sites[site].data['ZYXR']-1j*data_D.sites[site].data['ZYXI']
	zyyd = data_D.sites[site].data['ZYYR']-1j*data_D.sites[site].data['ZYYI']
	Z_D = np.array([[zxxd, zxyd], [zyxd, zyyd]])

	# zxx_ud = data_UD.sites[site].data['ZXXR']-1j*data_UD.sites[site].data['ZXXI']
	# zxy_ud = data_UD.sites[site].data['ZXYR']-1j*data_UD.sites[site].data['ZXYI']
	# zyx_ud = data_UD.sites[site].data['ZYXR']-1j*data_UD.sites[site].data['ZYXI']
	# zyy_ud = data_UD.sites[site].data['ZYYR']-1j*data_UD.sites[site].data['ZYYI']
	# Z_UD = np.array([[zxx_ud, zxy_ud], [zyx_ud, zyy_ud]])

	C = np.zeros((2,2,data_R.sites[site].NP))
	for ip in range(data_R.sites[site].NP):
		C[:,:,ip] = np.real(np.matmul(Z_D[:,:,ip], np.linalg.inv(Z_R[:,:,ip]))) # Calculate the distortion tensor
		all_C[:, ii, ip] = np.reshape(C[:,:,ip], [4,])
		Z_UD = np.matmul(C[:,:,ip], Z_R[:,:,ip])

		data_UD.sites[site].data['ZXXR'][ip] = np.real(Z_UD[0,0])
		data_UD.sites[site].data['ZXYR'][ip] = np.real(Z_UD[0,1])
		data_UD.sites[site].data['ZYXR'][ip] = np.real(Z_UD[1,0])
		data_UD.sites[site].data['ZYYR'][ip] = np.real(Z_UD[1,1])

		data_UD.sites[site].data['ZXXI'][ip] = -1*np.imag(Z_UD[0,0])
		data_UD.sites[site].data['ZXYI'][ip] = -1*np.imag(Z_UD[0,1])
		data_UD.sites[site].data['ZYXI'][ip] = -1*np.imag(Z_UD[1,0])
		data_UD.sites[site].data['ZYYI'][ip] = -1*np.imag(Z_UD[1,1])


np.save('E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/f-dependent_distortions', all_C)
data_UD.write('E:/phd/NextCloud/data/Regions/MetalEarth/larder/Hex2Mod/feature_tests/HexLL_base_rmDistortion_resp.dat')
# plt.plot(C[0,0,:], 'b.')
# plt.plot(C[0,1,:], 'r.')
# plt.plot(C[1,0,:], 'g.')
# plt.plot(C[1,1,:], 'k.')
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
