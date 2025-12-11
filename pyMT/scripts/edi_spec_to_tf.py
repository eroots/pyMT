import mtpy.core.mt as mt
import os
# Simple script to re-write spectra EDIs to transfer function EDIs
# Requires the older mtpy version (which requires numpy <= 1.21)

edi_path = 'E:/Work/sync/Regions/Reykjanes/edis/spectra/'
save_path = 'E:/Work/sync/Regions/Reykjanes/edis/'
# edi_files = ['L0.edi']


# bad_edis = ['L1000N_5900.edi', 'L1800N_5800.edi', 'L2600N_5400.edi', 
#             'L3400N_5400.edi', 'L4200N_5400.edi', 'L5000N_5400.edi',
#             'L5800N_7000.edi', 'L6600N_7000.edi', 'L7200N_7000.edi']
edi_files = os.listdir(edi_path)
edi_files = [x for x in edi_files if x.endswith('.edi')]
# edi_files = [x for x in edi_files if x not in bad_edis]

for f in edi_files:
    print(f)
    mt_obj = mt.MT(edi_path + f)
    # mt_obj.edi_obj.write_edi_file(save_path + f) # This one may be for mtpy-v2
    mt_obj.write_mt_file(save_dir=save_path,
                         fn_basename=f,
                         file_type='edi',
                         latlon_format='dms',
                         longitude_format='LONG',
                         new_Z_obj=mt_obj.Z,
                         new_Tipper_obj=mt_obj.Tipper)