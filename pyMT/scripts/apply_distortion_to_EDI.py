# import pyMT.data_structures as DS
import mtpy.core.mt as mt
import numpy as np
from copy import deepcopy
import os
import pyMT.IO


local_path = 'E:'
base_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/CART_test/j2/'
# data_file = 'study14_real-locs.dat'
list_file = 'allsites.lst'
# dummy_site_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/CART_test/j2/'
# dummy_sites = pyMT.IO.read_sites(dummy_site_path+'wst_cullmantle.lst')

# edi_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/'
# save_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/Edi_RotationFix/'

edi_path = local_path + '/phd/Nextcloud/data/Regions/MetalEarth/CART_test/j2/'
save_path1 = edi_path + 'distortion/'
save_path2 = edi_path + 'distortion-staticOnly/'
# Uncomment next line to run it over all EDIs in the folder
edi_files = [x for x in os.listdir(edi_path) if x.endswith('edi')]
write_fixed_edis = True

for ii, file in enumerate(edi_files):
	# mt_obj_dummy = mt.MT(fn=dummy_site_path+dummy_sites[ii]+'.edi')
	mt_obj = mt.MT(fn=edi_path + file)
	C_diag = np.random.uniform(size=2, low=0, high=2)
	C_offdiag = np.random.uniform(size=2, low=0, high=0.5)
	C = np.array([[C_diag[0], C_offdiag[0]], [C_offdiag[1], C_diag[1]]])
	D = np.linalg.inv(C)
	D_inv, new_Z, Z_err = mt_obj.Z.remove_distortion(distortion_tensor=D)
	mt_obj.Z.z = new_Z
	if write_fixed_edis:
		mt_obj.write_mt_file(save_dir=save_path1,
							 fn_basename=file,
						 	 longitude_format='LONG')
	mt_obj = mt.MT(fn=edi_path + file)
	C = np.array([[C_diag[0], 0], [0, C_diag[1]]])
	D = np.linalg.inv(C)
	D_inv, new_Z, Z_err = mt_obj.Z.remove_distortion(distortion_tensor=D)
	mt_obj.Z.z = new_Z
	if write_fixed_edis:
		mt_obj.write_mt_file(save_dir=save_path2,
							 fn_basename=file,
						 	 longitude_format='LONG')

# plt.scatter(lons, lats, c=median_strike)
# for ii in range(len(edi_files)):
# 	plt.text(x=lons[ii], y=lats[ii], s='{:3.2f}'.format(median_strike[ii]))
# plt.show()






# Using pyMT
# outfile = 'study14_real-locs_fulldistorted.dat'
# outfile2 = 'study14_real-locs_staticdistorted.dat'
# data = DS.Data(base_path + data_file)
# data2 = deepcopy(data)
# for site in data.site_names:
# 	# Try to make sure at least the 'phase-mixing' term
# 	C_diag = np.random.uniform(size=2, low=0, high=2)
# 	C_offdiag = np.random.uniform(size=2, low=0, high=0.5)
# 	C = np.array([[C_diag[0], C_offdiag[0]], [C_offdiag[1], C_diag[1]]])
# 	# Assume static shifts only - no phase mixing (see Jones and Chave, page 235-236)
# 	# C[0,1] = 0
# 	# C[1,0] = 0
# 	for ii in range(data.NP):
# 		zxy = data.sites[site].data['ZXYR'][ii] + 1j*data.sites[site].data['ZXYI'][ii]
# 		zyx = data.sites[site].data['ZYXR'][ii] + 1j*data.sites[site].data['ZYXI'][ii]
# 		zxx = data.sites[site].data['ZXXR'][ii] + 1j*data.sites[site].data['ZXXI'][ii]
# 		zyy = data.sites[site].data['ZYYR'][ii] + 1j*data.sites[site].data['ZYYI'][ii]
# 		Z = np.array(([zxx,zxy],[zyx,zyy]))	
# 		Zd = np.matmul(C,Z)	
# 		data.sites[site].data['ZXXR'][ii] = np.real(Zd[0,0])
# 		data.sites[site].data['ZXYR'][ii] = np.real(Zd[0,1])
# 		data.sites[site].data['ZYXR'][ii] = np.real(Zd[1,0])
# 		data.sites[site].data['ZYYR'][ii] = np.real(Zd[1,1])
# 		data.sites[site].data['ZXXI'][ii] = np.imag(Zd[0,0])
# 		data.sites[site].data['ZXYI'][ii] = np.imag(Zd[0,1])
# 		data.sites[site].data['ZYXI'][ii] = np.imag(Zd[1,0])
# 		data.sites[site].data['ZYYI'][ii] = np.imag(Zd[1,1])

# 		C_static = np.array([[C[0,0], 0], [0, C[1,1]]])
# 		Zd = np.matmul(C_static, Z)	
# 		data2.sites[site].data['ZXXR'][ii] = np.real(Zd[0,0])
# 		data2.sites[site].data['ZXYR'][ii] = np.real(Zd[0,1])
# 		data2.sites[site].data['ZYXR'][ii] = np.real(Zd[1,0])
# 		data2.sites[site].data['ZYYR'][ii] = np.real(Zd[1,1])
# 		data2.sites[site].data['ZXXI'][ii] = np.imag(Zd[0,0])
# 		data2.sites[site].data['ZXYI'][ii] = np.imag(Zd[0,1])
# 		data2.sites[site].data['ZYXI'][ii] = np.imag(Zd[1,0])
# 		data2.sites[site].data['ZYYI'][ii] = np.imag(Zd[1,1])

# data.write(base_path + outfile)
# data2.write(base_path + outfile2)

