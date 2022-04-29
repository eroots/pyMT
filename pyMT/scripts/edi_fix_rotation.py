# from mtpy.core.mt import MT
import matplotlib.pyplot as plt
import mtpy.core.mt as mt
import numpy as np
import os

# Lalor interp freqs
interp_freqs = [1.500000E+05, 1.207725E+05, 9.724000E+04, 7.829279E+04, 6.303745E+04, 5.075461E+04, 
 			   4.086508E+04, 3.290252E+04, 2.649147E+04, 2.132961E+04, 1.717353E+04, 1.382727E+04,
 			   1.113303E+04, 8.963762E+03, 7.217174E+03, 5.810908E+03, 4.678653E+03, 3.767018E+03,
 			   3.033015E+03, 2.442032E+03, 1.966202E+03, 1.583088E+03, 1.274623E+03, 1.026263E+03,
 			   8.262961E+02, 6.652924E+02, 5.356602E+02, 4.312869E+02, 3.472507E+02, 2.795889E+02, 
 			   2.251110E+02, 1.812482E+02, 1.459320E+02, 1.174971E+02, 9.460286E+01, 7.616950E+01,
 			   6.132788E+01, 4.937815E+01, 3.975682E+01, 3.201021E+01, 2.577302E+01, 2.075115E+01,
 			   1.670779E+01, 1.345228E+01, 1.083110E+01, 8.720666E+00, 7.021445E+00, 5.653317E+00,
 			   4.551769E+00, 3.664857E+00, 2.950760E+00, 2.375804E+00, 1.912879E+00, 1.540155E+00,
 			   1.240056E+00, 9.984313E-01, 8.038870E-01, 6.472497E-01, 5.211332E-01, 4.195904E-01,
 			   3.378333E-01, 2.720065E-01, 2.190060E-01, 1.763327E-01, 1.419743E-01, 1.143106E-01,
 			   9.203724E-02, 7.410379E-02, 5.966467E-02, 4.803902E-02, 3.867862E-02, 3.114209E-02,
 			   2.507406E-02, 2.018838E-02, 1.625467E-02, 1.308745E-02, 1.053736E-02, 8.484162E-03, 
 			   6.831024E-03, 5.500000E-03,]

# edi_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/'
# save_path = 'E:/phd/NextCloud/data/Regions/snorcle/j2/2020-collation-ian/fixrot/Edi_RotationFix/'

# edi_path = 'E:/phd/NextCloud/data/Regions/plc18/PLC MT-20210303T165501Z-001/PLC MT/edi/'
# edi_path = 'E:/phd/NextCloud/data/Regions/thot/j2/spectra/'
# edi_path = 'E:/phd/NextCloud/data/Regions/lalor/j2/from_Masoud/dat_files/j2edi/fix/'
# edi_path = 'E:/phd/NextCloud/data/Regions/lalor/j2/from_Masoud/geotools_rotated/L192/fix/'
edi_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/red_lake/j2/original/MT/'
save_path = 'E:/phd/NextCloud/data/Regions/MetalEarth/red_lake/j2/bbmt/' #+ 'mtpy_rotate/'
rotate_by = 0
zero_azimuth = True
scrub_rotation = False
interp_data = False
# edi_files = ['plc002.edi']
# Uncomment next line to run it over all EDIs in the folder
edi_files = [x for x in os.listdir(edi_path) if x.endswith('edi')]
write_fixed_edis = False

non_uniform = []
bad_sites = []
for ii, file in enumerate(edi_files):
	try:
		mt_obj = mt.MT(edi_path + file)
	except ValueError:
		bad_sites.append(file)
		continue
	if ~np.all(mt_obj.Z.rotation_angle == mt_obj.Z.rotation_angle[0]):
		non_uniform.append(file)
	if write_fixed_edis:
		# mt_obj.Z.rotation_angle = np.zeros(mt_obj.Z.rotation_angle.shape) + rotate_by
		if rotate_by:
			mt_obj.Z.rotate(rotate_by)
		elif scrub_rotation:
			# mt_obj.Z.rotation_angle = 0
			mt_obj.Z.rotation_angle = np.zeros(mt_obj.Z.rotation_angle.shape)
		elif zero_azimuth:
			mt_obj.Z.rotate(-1*mt_obj.Z.rotation_angle[0])
			mt_obj.Tipper.rotate(-1*mt_obj.Tipper.rotation_angle[0])
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
			mt_obj.write_mt_file(save_dir=save_path,
								 fn_basename=file,
								 latlon_format='dms',
								 longitude_format='LONG')


plt.scatter(lons, lats, c=median_strike)
for ii in range(len(edi_files)):
	plt.text(x=lons[ii], y=lats[ii], s='{:3.2f}'.format(median_strike[ii]))
plt.show()

