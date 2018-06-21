import mtpy.core.mt as mt
import os

# in_path = 'C:/Users/eric/phd/ownCloud/Metal Earth/Data/PROCESSED DATA/'
in_path = 'F:/Dropbox/PROCESSED DATA/'
# out_path = 'C:/Users/eric/phd/ownCloud/Metal Earth/Data/PROCESSED DATA/Converted Edi/'
out_path = 'F:/PROCESSED DATA/Converted Edi/'
extended_path = ['ATIKOKAN/EDIs', 'DRYDEN/EDIs', 'RAINY RIVER/EDIs', 'STURGEON LAKES/EDIs']
for transect in extended_path:
    full_path = ''.join([in_path, transect])
    for file in os.listdir(full_path):
        if file.endswith('.edi'):
            if file not in os.listdir(out_path):
                mt_obj = mt.MT('/'.join([full_path, file]), data_type='spectra')
                mt_obj.write_edi_file(''.join([out_path, file]))
