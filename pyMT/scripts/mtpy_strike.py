from mtpy.analysis.geometry import strike_angle, dimensionality
from mtpy.core.mt import MT
from mtpy.core.z import Z
import numpy as np
import os
import matplotlib.pyplot as plt
import mtpy.utils.gis_tools as gis_tools

# edi_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/R2Southeast'
# save_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/swayze/j2/R2Southeast/strike_rotated/'
# edi_path = 'E:/phd/NextCloud/data/Regions/thot/j2/line_L/'
# save_path = 'E:/phd/NextCloud/data/Regions/thot/j2/line_L/strike_rotated/'
edi_path = 'E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/all/'
save_path = 'E:/phd/NextCloud/VirtualBoxShare/2dAni/j2/1d/selected/'
edi_files = [x for x in os.listdir(edi_path) if x.endswith('edi')]
low_cut = 0.0001 # in Hz
high_cut = 1
rotate_to_strike = 0
desired_strike = 30
# include_list = ['site145.edi', 'WST27.edi', 'WST85.edi', 'WST79.edi', 'WST76.edi', 'WST88.edi', 'site215.edi']
include_list = ['USArray.NDE32.2017.edi', 'USArray.MNC33.2012.edi', 'USArray.MNC32.2017.edi',
				'98-1_056.edi', '98-1_053.edi', '98-1_030.edi', '98-1_080.edi']
median_strike, lats, lons = [], [], []
for ii, file in enumerate(edi_files):
	if file in include_list:
		mt_obj = MT(edi_path + file)
		strike = strike_angle(z_object=mt_obj.Z,
							  skew_threshold=5,
							  eccentricity_threshold=0.1)
		dim = dimensionality(z_object=mt_obj.Z,
							 skew_threshold=5,
							 eccentricity_threshold=0.1)
		freq_idx = (mt_obj.Z.freq > low_cut) + (mt_obj.Z.freq < high_cut)
		freq_idx = freq_idx * dim < 3
		med = np.nanmedian(strike[freq_idx,0])
		median_strike.append(med)
		lats.append(mt_obj.lat)
		lons.append(mt_obj.lon)
		if rotate_to_strike:
			epsg = gis_tools.get_epsg(mt_obj.lat, mt_obj.lon)
			utm = gis_tools.get_utm_zone(mt_obj.lat, mt_obj.lon)
			# print(utm)
			utm_x, utm_y = gis_tools.project_point_ll2utm(mt_obj.lat, mt_obj.lon, utm_zone=utm[0])[:2]
			# mask = (dim < 3) * (abs(strike[:,0] - desired_strike) < 10) * freq_idx
			mask = (dim < 3)
			new_obj = Z(z_array=mt_obj.Z.z[mask],
						z_err_array=mt_obj.Z.z_err[mask],
						freq=mt_obj.Z.freq[mask])
			# new_obj.rotate(desired_strike)
			new_obj.rotate(med)
			new_file_name = file[:-4]+'_rot.edi'
			mt_obj.write_mt_file(save_dir=save_path,
								 fn_basename=new_file_name,
								 new_Z_obj=new_obj,
								 latlon_format='dd')
			with open(save_path + 'coordinates.txt', 'a') as f:
				f.write('{}_{:>10s} {:>10.2f} {:>10.2f} {:>5.2f}\n'.format(ii, new_file_name, utm_x, utm_y, mt_obj.elev))


plt.scatter(lons, lats, c=median_strike)
for ii in range(len(lons)):
	plt.text(x=lons[ii], y=lats[ii], s='{:3.2f}'.format(median_strike[ii]))
plt.show()

