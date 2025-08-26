# from mtpy.core.mt import MT
# import matplotlib.pyplot as plt
import mtpy.core.mt as mt
import mtpy.core.edi as edi
import numpy as np
import os

# Lalor interp freqs
# interp_freqs = [1.500000E+05, 1.207725E+05, 9.724000E+04, 7.829279E+04, 6.303745E+04, 5.075461E+04, 
#  			   4.086508E+04, 3.290252E+04, 2.649147E+04, 2.132961E+04, 1.717353E+04, 1.382727E+04,
#  			   1.113303E+04, 8.963762E+03, 7.217174E+03, 5.810908E+03, 4.678653E+03, 3.767018E+03,
#  			   3.033015E+03, 2.442032E+03, 1.966202E+03, 1.583088E+03, 1.274623E+03, 1.026263E+03,
#  			   8.262961E+02, 6.652924E+02, 5.356602E+02, 4.312869E+02, 3.472507E+02, 2.795889E+02, 
#  			   2.251110E+02, 1.812482E+02, 1.459320E+02, 1.174971E+02, 9.460286E+01, 7.616950E+01,
#  			   6.132788E+01, 4.937815E+01, 3.975682E+01, 3.201021E+01, 2.577302E+01, 2.075115E+01,
#  			   1.670779E+01, 1.345228E+01, 1.083110E+01, 8.720666E+00, 7.021445E+00, 5.653317E+00,
#  			   4.551769E+00, 3.664857E+00, 2.950760E+00, 2.375804E+00, 1.912879E+00, 1.540155E+00,
#  			   1.240056E+00, 9.984313E-01, 8.038870E-01, 6.472497E-01, 5.211332E-01, 4.195904E-01,
#  			   3.378333E-01, 2.720065E-01, 2.190060E-01, 1.763327E-01, 1.419743E-01, 1.143106E-01,
#  			   9.203724E-02, 7.410379E-02, 5.966467E-02, 4.803902E-02, 3.867862E-02, 3.114209E-02,
#  			   2.507406E-02, 2.018838E-02, 1.625467E-02, 1.308745E-02, 1.053736E-02, 8.484162E-03, 
#  			   6.831024E-03, 5.500000E-03,]
# Kaapvaal interp freqs
# interp_freqs = [3.600000e+02, 3.000000e+02, 2.600000e+02, 2.200000e+02, 1.800000e+02, 1.500000e+02,
# 			    1.300000e+02, 1.100000e+02, 9.000000e+01, 7.500000e+01, 6.500000e+01, 5.500000e+01,
# 			    4.500000e+01, 3.700000e+01, 3.300000e+01, 2.750000e+01, 2.250000e+01, 1.880000e+01,
# 			    1.620000e+01, 1.370000e+01, 1.120000e+01, 9.400000e+00, 8.100000e+00, 6.900000e+00,
# 			    5.600000e+00, 4.900000e+00, 4.100000e+00, 3.400000e+00, 2.810000e+00, 2.440000e+00,
# 			    2.060000e+00, 1.690000e+00, 1.410000e+00, 1.220000e+00, 1.030000e+00, 8.400000e-01,
# 			    7.000000e-01, 6.100000e-01, 5.200000e-01, 4.200000e-01, 3.500000e-01, 3.050000e-01,
# 			    2.580000e-01, 2.110000e-01, 1.760000e-01, 1.520000e-01, 1.290000e-01, 1.050000e-01,
# 			    8.800000e-02, 7.600000e-02, 6.400000e-02, 5.300000e-02, 4.400000e-02, 4.000000e-02,
# 			    3.800000e-02, 3.200000e-02, 3.000000e-02, 2.640000e-02, 2.200000e-02, 2.000000e-02,
# 			    1.900000e-02, 1.610000e-02, 1.500000e-02, 1.320000e-02, 1.100000e-02, 1.000000e-02,
# 			    9.500000e-03, 8.100000e-03, 7.500000e-03, 6.600000e-03, 5.500000e-03, 5.000000e-03,
# 			    4.800000e-03, 4.000000e-03, 3.750000e-03, 3.300000e-03, 2.750000e-03, 2.500000e-03,
# 			    2.380000e-03, 2.010000e-03, 1.875000e-03, 1.650000e-03, 1.370000e-03, 1.250000e-03,
# 			    1.190000e-03, 1.010000e-03, 9.375000e-04, 8.200000e-04, 6.900000e-04, 6.250000e-04,
# 			    6.000000e-04, 5.000000e-04, 4.687500e-04, 4.100000e-04, 3.400000e-04, 3.125000e-04,
# 			    2.980000e-04, 2.520000e-04, 2.343750e-04, 2.060000e-04, 1.720000e-04, 1.562500e-04,
# 			    1.490000e-04, 1.260000e-04, 1.171870e-04, 1.030000e-04, 7.812500e-05, 5.859370e-05]

# mt_obj = edi.Edi('G:/Other computers/My Computer/sync/Regions/undercover/j2/UND-rotated/UND-001.edi')
# interp_freqs = mt_obj.Z.freq
# edi_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/'
# save_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/Edi_RotationFix/'

# edi_path = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/edi/'
# edi_path = 'E:/phd/NextCloud/data/Regions/thot/j2/spectra/'
# edi_path = 'E:/phd/NextCloud/data/Regions/lalor/j2/from_Masoud/dat_files/j2edi/fix/'
# edi_path = 'E:/phd/NextCloud/data/Regions/lalor/j2/from_Masoud/geotools_rotated/L192/fix/'
# edi_path = 'E:/phd/NextCloud/data/Regions/samtex/j2/edis/'

# edi_path  = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/Koillismaa_DeepHole_MT_unrotated/from-FFMT/'
# edi_path = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/Koillismaa_DeepHole_MT_declRemoved/'
# save_path = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/Koillismaa_DeepHole_MT_declRemoved/rot-scrubbed/'

# edi_path  = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/UND-wDeclination/'
# save_path = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/UND-rotated/'

edi_path = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/UND-rotated/'
save_path = 'G:/Other computers/My Computer/sync/Regions/undercover/j2/UND-rotated/rot-scrubbed/'

# edi_path = 'E:/phd/NextCloud/data/Regions/NACP/j2/fromJim/'
# save_path = 'E:/phd/NextCloud/data/Regions/NACP/j2/fromJim/mtpy/'
# edi_path = 'E:/phd/NextCloud/data/Regions/churchill/j2/'
# save_path = 'E:/phd/NextCloud/data/Regions/churchill/j2/mtpy/zero_azimuth/'

rotate_by = 0
zero_azimuth = False
scrub_rotation = True
interp_data = False
# edi_files = ['plc002.edi']
# Uncomment next line to run it over all EDIs in the folder
edi_files = [x for x in os.listdir(edi_path) if x.endswith('edi')]
write_fixed_edis = True

non_uniform = []
bad_sites = []
for ii, file in enumerate(edi_files):
	try:
		# mt_obj = mt.MT(edi_path + file)
		mt_obj = edi.Edi(edi_path + file)
	except:
		bad_sites.append(file)
		print('Problem reading site: {}'.format(file))
		continue
	if ~np.all(mt_obj.Z.rotation_angle == mt_obj.Z.rotation_angle[0]):
		non_uniform.append(file)
	
		# mt_obj.Z.rotation_angle = np.zeros(mt_obj.Z.rotation_angle.shape) + rotate_by
	if rotate_by:
		mt_obj.Z.rotate(rotate_by)
		mt_obj.Tipper.rotate(rotate_by)
	elif scrub_rotation:
		# mt_obj.Z.rotation_angle = 0
		mt_obj.Z.rotation_angle = np.zeros(mt_obj.Z.rotation_angle.shape)
		mt_obj.Tipper.rotation_angle = np.zeros(mt_obj.Tipper.rotation_angle.shape)
	elif zero_azimuth:
		mt_obj.Z.rotate(-1*mt_obj.Z.rotation_angle[0])
		mt_obj.Tipper.rotate(-1*mt_obj.Tipper.rotation_angle[0])
	if write_fixed_edis:
		if interp_data:
			use_interp_freqs = np.array(interp_freqs)
			idx = np.where((use_interp_freqs > min(mt_obj.Z.freq)) & (use_interp_freqs < max(mt_obj.Z.freq)))
			use_interp_freqs = use_interp_freqs[idx]
			# use_interp_freqs[use_interp_freqs < min(mt_obj.Z.freq)] = []
			mt_z, mt_k = mt_obj.interpolate(use_interp_freqs)
			mt_obj.write_mt_file(save_dir=save_path,
							      fn_basename=file,
							      latlon_format='dms',
							      longitude_format='LONG',
							      new_Z_obj=mt_z, new_Tipper_obj=mt_k)
		else:
		# mt_obj.write_mt_file(save_dir=save_path,
			# 					 fn_basename=file,
			# 					 latlon_format='dms',
			# 					 longitude_format='LONG')
			try:
				mt_obj.write_edi_file(new_edi_fn=save_path+file,
									  latlon_format='dms',
								      longitude_format='LONG')
			except:
				print('Problem writing site: {}'.format(file))
				bad_sites.append(file)


# plt.scatter(lons, lats, c=median_strike)
# for ii in range(len(edi_files)):
# 	plt.text(x=lons[ii], y=lats[ii], s='{:3.2f}'.format(median_strike[ii]))
# plt.show()

