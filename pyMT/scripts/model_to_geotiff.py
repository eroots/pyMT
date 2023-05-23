import pyMT.data_structures as DS
import pyMT.IO as IO
import pyMT.utils as utils
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.interpolate import RegularGridInterpolator

# This function now defined as a method within Model objects
def average_conductance(model, idx_0, idx_1):
	conductance = np.zeros((model.nx, model.ny))
	for ii in range(idx_0, idx_1):
		conductance += model.zCS[ii] * (1 / model.vals[:, :, ii])
	return conductance

# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/smooth2/wstZK-s2_lastIter.rho')
# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/cull/Z/ZK/wstZK_lastIter.rho')
model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/ger_newSites/wst2ger/wst2ger-all-fromCapped_lastIter.rho')
data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/j2/ger_2020R1.lst')
# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/matheson/Hex2Mod/HexMat_Z.model')
# data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
# site_data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/matheson/j2/MATall.lst')
# outfile = 'E:/phd/NextCloud/data/ArcMap/A-G/models/matheson_MT-Model.tiff'
# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/ger_newSites/for_ade/hs1000/s2/gerS2-all_lastIter.rho')
# data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/geraldton/ger_newSites/for_ade/EDIs/ger_2020R1-2.lst')
# outfile = 'E:/phd/NextCloud/data/ArcMap/WST/models/wst2dry-from_cullZK_capped10000.tiff'
# data.locations = data.get_locs(mode='lambert')
# model = DS.Model('E:/phd/NextCloud/data/Regions/MetalEarth/dryden/wst2dry2/smooth2/capped10000/from-cullZK/wst2dry-capped_lastIter.rho')
# data = DS.RawData('E:/phd/NextCloud/data/Regions/MetalEarth/dryden/j2/dry_noOOQ.lst')
# outfile = 'E:/phd/NextCloud/data/ArcMap/A-G/models/HexAG_Z_static_UTM17.tiff'
# model = DS.Model('E:/phd/NextCloud/data/Regions/plc18/final/plc31-2_NLCG_125.rho')
# data = DS.RawData('E:/phd/NextCloud/data/Regions/plc18/j2/all.lst')
# outfile = 'E:/phd/NextCloud/data/ArcMap/plc/plc-final_UTM12.tiff'
data.locations = data.get_locs(mode='lambert')
# data.to_utm(17, 'N')

# # Remove the pads so you don't have a million cells
# for ii in range(20):
# 	model.dx_delete(0)
# 	model.dx_delete(-1)
# 	model.dy_delete(0)
# 	model.dy_delete(-1)
# 	# model.dz_delete(-1) # If you want

model.origin = data.origin
model.to_lambert()
# model.to_UTM()

x, y, z = (utils.edge2center(model.dy),
		   utils.edge2center(model.dx),
		   utils.edge2center(model.dz))
X, Y, Z =  np.meshgrid(x, y, z, indexing='ij', sparse=True)
# points = np.array((X.flatten(), Y.flatten())).T
# Z = np.array([[iz] * X.size for iz in model.dz])

xCS = np.min(model.xCS)
yCS = np.min(model.yCS)

x_reg, y_reg = (np.linspace(x[0], x[-1], int((x[-1] - x[0]) / yCS)),
				np.linspace(y[0], y[-1], int((y[-1] - y[0]) / xCS)))
qx, qy = np.meshgrid(x_reg, y_reg)


transform = Affine.translation(x[0], y[0]) * Affine.scale(np.min(model.yCS), np.min(model.xCS))

for kind in ['X', 'Y', 'X-Y'][0:1]:
	if kind == 'X':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_x.transpose([1, 0, 2]), method='nearest')
	elif kind == 'Y':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_y.transpose([1, 0, 2]), method='nearest')
	elif kind == 'X-Y':
		RGI = RegularGridInterpolator((x, y, z),
								  model.rho_x.transpose([1, 0, 2]) / model.rho_y.transpose([1, 0, 2]), method='nearest')
	for iz in range(model.nz):
		# outfile = 'E:/phd/NextCloud/data/ArcMap/WST/models/wstAni_separate/{}/wstAni{}_{:5.2f}km.tiff'.format(kind, kind,z[iz] / 1000)
		outfile = 'E:/phd/NextCloud/data/ArcMap/WST/models/wst2ger/wst2gerZK_{:5.2f}km.tiff'.format(z[iz] / 1000)
		with rasterio.open(outfile,
							'w', driver='GTiff',
							height=y_reg.size, width=x_reg.size, count=1,
							dtype=np.float64,
							crs={'init': 'EPSG:3979'}, # Lambert
							# crs={'init': 'EPSG:32617'}, # UTM zone (last 2 digits)
							transform=transform) as dst:
			dst.set_band_description(1, 'depth = {:5.2f} km'.format(z[iz]/1000))
			query_points = np.zeros((qx.size, 3))
			query_points[:, 0] = qx.flatten()
			query_points[:, 1] = qy.flatten()
			query_points[:, 2] = z[iz]
			vals = RGI(query_points)
			dst.write(np.log10(vals.reshape([qx.shape[0], qx.shape[1]])), 1)

# # Uncomment and shift indents for a single large geotiff
# with rasterio.open(outfile,
# 					'w', driver='GTiff',
# 					height=y_reg.size, width=x_reg.size, count=17, #model.nz,
# 					dtype=np.float64,
# 					# crs={'init': 'EPSG:3979'}, # Lambert
# 					crs={'init': 'EPSG:32617'}, # UTM zone (last 2 digits)
# 					transform=transform) as dst:
# 	for iz in range(17):
# 		# dst.update_tags(iz + 1, depth=int(z[iz]))
# 		dst.set_band_description(iz + 1, 'depth = {:5.2f} km'.format(z[iz]/1000))
# 		query_points = np.zeros((qx.size, 3))
# 		query_points[:, 0] = qx.flatten()
# 		query_points[:, 1] = qy.flatten()
# 		query_points[:, 2] = z[iz]
# 		vals = RGI(query_points)
# 		dst.write(np.log10(vals.reshape([qx.shape[0], qx.shape[1]])), iz + 1)

	
