import pyMT.data_structures as DS
import numpy as np
import pyproj
from pyMT.IO import verify_input
import pyMT.utils as utils

def transform_locations(dataset, UTM):
    dataset.raw_data.locations = dataset.raw_data.get_locs(mode='latlong')
    if UTM.lower() == 'lam':
        transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3979')
        for ii, (lat, lon) in enumerate(dataset.raw_data.locations):
            x, y = transformer.transform(lat, lon)
            dataset.raw_data.locations[ii, :] = y, x

    else:
        UTM_letter = UTM[-1]
        while True:
            if len(UTM) == 3:
                UTM_number = int(UTM[:2])
                break
            elif len(UTM) == 2:
                UTM_number = int(UTM[0])
                break
            else:
                print('UTM {} is not a valid zone.'.format(UTM))
                UTM = verify_input('UTM Zone',
                                   expected=str,
                                   default='dummy')
        for ii in range(len(dataset.raw_data.locations)):
            easting, northing = project((dataset.raw_data.locations[ii, 1],
                                         dataset.raw_data.locations[ii, 0]),
                                        zone=UTM_number, letter=UTM_letter)[2:]
            dataset.raw_data.locations[ii, 1], dataset.raw_data.locations[ii, 0] = easting, northing


list_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/wst_cullmantle.lst'
# data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_{}_flagged.dat'.format('all')
data_file = 'E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle3_K.dat'
dataset = DS.Dataset(listfile=list_file, datafile=data_file)
transform_locations(dataset, 'lam')
# for file in ('all', 'XY', 'YX', 'K'):
# dataset.data = DS.Data(datafile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_base_K.dat'.format(file))
dataset.data = DS.Data(datafile='E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle3_K.dat')
for ii, site in enumerate(dataset.raw_data.site_names):
    dataset.data.sites[site].locations['X'], dataset.data.sites[site].locations['Y'] = dataset.raw_data.locations[ii, :]
dataset.data.locations = utils.center_locs(dataset.data.get_locs())[0]
# dataset.data.write('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle_LAMBERT_K_flagged.dat'.format(file))
dataset.data.write('E:/phd/NextCloud/data/Regions/MetalEarth/wst/fullmantle/wst_cullmantle3_LAMBERT_K_flagged.dat')