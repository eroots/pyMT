import pyMT.data_structures as DS
import pyMT.utils as utils
import numpy as np
from copy import deepcopy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator as RGI
#########
# Depth dependent painting
# depths = [500, 1000, 1500, 2000, 2500, 3500, 5000, 7000, 10000, 15000, 20000, 25000, 35000]
# # depths = [500]
# orig_model = DS.Model('E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/plc18_NLCG_116.rho')

# for d in depths:
# 	outfile = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/tests/plc_rotatedTest-{}m.rho'.format(d)
# 	model = deepcopy(orig_model)
# 	z1 = np.argmin(abs(d - np.array(model.dz)))
# 	z2 = 49
# 	cc = 0
# 	for xx in range(42, 74):
		
# 		y1 = int(57 + np.floor(cc / 2))
# 		y2 = int(83 - 10 + np.floor(cc / 2))
# 		cc += 1
# 		for yy in range(y1, y2):
# 			for zz in range(z1, z2):
# 				if model.vals[xx, yy, zz] > 1000:
# 					model.vals[xx, yy, zz] = 1000
# 	model.write(outfile)


# orig_model = DS.Model('E:/phd/NextCloud/data/Regions/plc18/final/plc31-2_NLCG_125.rho')
orig_model = DS.Model('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/wstZK_s1_nest.mod', file_format='em3dani')
model2 = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/mantle/wNorth/embed_shield/hs300-1000/wst_hs300-1000_lastIter.rho')
model = deepcopy(orig_model)
ix1, ix2 = 0, 60
iy1, iy2 = 0, 70
x = utils.edge2center(model2.dx)
y = utils.edge2center(model2.dy)
z = utils.edge2center(model2.dz)
interpolator = RGI((y, x, z), np.transpose(model2.vals,
				   [1, 0, 2]), bounds_error=False, fill_value=5)
qx, qy = np.meshgrid(utils.edge2center(model.dx[ix1:ix2+1]),
					 utils.edge2center(model.dy[iy1:iy2+1]))
qx = qx.flatten()
qy = qy.flatten()
z_interp = utils.edge2center(model.dz)
for iz in range(55,65):
	qz = np.zeros(qx.shape) + z_interp[iz+2]
	query_points = np.zeros((len(qx), 3))
	query_points[:, 0] = qy
	query_points[:, 1] = qx
	query_points[:, 2] = qz
	points = interpolator(query_points)
	model.vals[ix1:ix2,iy1:iy2,iz] = np.reshape(points, [iy2-iy1, ix2-ix1]).T
model.vals[0:70, 0:75, 54:66] = gaussian_filter(model.vals[0:70, 0:75, 54:66], [1.,1.,1.])
model.write('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/wstZK_s1_nest-tailDykes2', file_format='em3dani')
# rho_dyke = 10
# for iz in range(55, 62):	
# 	for iy in range(1, 13):
# 		cc = 1
# 		for ix in range(32, 65):
# 			model.vals[ix, iy+cc, iz] = rho_dyke
# 			cc += 1
# for iz in range(55, 62):	
# 	for iy in range(23, 40):
# 		cc = 1
# 		for ix in range(32, 65):
# 			model.vals[ix, iy+cc, iz] = rho_dyke
# 			cc += 1
# for iz in range(55, 62):	
# 	for iy in range(2, 40):
# 		cc = 1
# 		for ix in range(32, 65):
# 			if model.vals[ix,iy+cc,iz] != rho_dyke:
# 				model.vals[ix, iy+cc, iz] = 5000
# 			cc += 1
# model.vals[20:85, 0:80, 54:63] = gaussian_filter(model.vals[20:85, 0:80, 54:63], [1.,1.,1.])
# model.write('E:/phd/NextCloud/data/synthetics/EM3DANI/wst/wstZK_s1_nest-tailDykes', file_format='em3dani')

# # thresh_rho = [300, 500, 1000]
# # tags = ['C1', 'C2']
# # x = [[52, 82], [40, 56]]
# # y = [[46, 66], [78, 97]]
# # z = [[43, 55], [18,43]]
# # tags = ['MC1', 'MC2', 'MC3', 'MC4']
# # x = [[81, 175], [41, 101], [49, 109], [74, 154]]
# # y = [[19, 49], [70, 104], [119, 154], [184, 215]]
# # z = [[23, 28], [23,28], [23,28], [23,28]]
# # tags = ['LC1', 'LC2']
# # thresh_rho = [1000]
# # x = [[26, 106], [95, 193]]
# # y = [[26, 155], [56, 215]]
# # z = [[27, 32], [27,32]]
# # thresh_rho = [100, 300, 500, 1000]
# # tags = ['R1', 'R2']
# # x = [[26, 116], [26, 106]]
# # y = [[76, 141], [141, 186]]
# # z = [[23, 28], [23,28]]
# tags = ['R1']
# thresh_rho = [500, 1000, 2500]
# x = [[47, 84]]
# y = [[58, 89]]
# z = [[24, 48]]
# for thresh in thresh_rho[:1]:
# 	# model = deepcopy(orig_model)
# 	for ii in range(len(x)):
# 		x1, x2 = x[ii]
# 		y1, y2 = y[ii]
# 		z1, z2 = z[ii]
# 		model = deepcopy(orig_model)
# 		for ix in range(x1, x2):
# 			for iy in range(y1, y2):
# 				for iz in range(z1, z2):
# 					if model.vals[ix,iy,iz] > thresh:
# 					# if model.vals[ix,iy,iz] < thresh:
# 						## LC1 refine
# 						# if not ((iy < 104) and (ix > 87)):
# 						## C2 refine
# 						# if not ((iz < 32) and (iy > 93) and (ix > 47)):
# 						## R refine
# 						if not ((iy > 83) and (ix < 68)):
# 							model.vals[ix,iy,iz] = thresh
# 		model.write('E:/phd/NextCloud/data/Regions/plc18/final/feature_test/{}-{}ohm'.format(tags[ii], thresh))
# 	# model.write('E:/phd/NextCloud/data/Regions/plc18/final/feature_test/AllCs-{}ohm'.format(thresh))