import simplekml
import pyMT.data_structures as WSDS
import pandas as pd
import numpy as np
import pyMT.utils as utils
from pyproj import Proj

# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/wst/New/j2/all.lst'
# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/abi-gren/New/j2/allsites.lst'
# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/dryden/j2/allsites.lst'
# list_file = r'C:/Users/eric/Documents/MATLAB/MATLAB/Inversion/Regions/MetalEarth/swayze/j2/SWZall.lst')
transects = ('attikokan',
             'chibougamau',
             'dryden',
             'geraldton',
             'larder',
             'malartic',
             'matheson',
             'rainy',
             'rouyn',
             'sturgeon',
             'swayze')
lists = ('ATTall.lst',
         'CHIall.lst',
         'DRYall.lst',
         'GERall.lst',
         'LARall.lst',
         'MALall.lst',
         'MATall.lst',
         'RRVall.lst',
         'ROUall.lst',
         'STUall.lst',
         'SWZall.lst')
list_path = 'C:/Users/eric/phd/ownCloud/Metal Earth/Data/ConvertedEDIs/FinalEDIs/'
kml_save_path = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/MT-locations/KMLs/'
csv_save_path = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/MT-locations/CSVs/'
write_kml = True
write_csv = False
UTM = 17
for ii, lst in enumerate(lists):
    list_file = ''.join([list_path, lst])
    kml_save_file = ''.join([kml_save_path, transects[ii], '.kml'])
    csv_save_file = ''.join([csv_save_path, transects[ii]])
    data = WSDS.RawData(listfile=list_file)
    # data = pd.read_table('C:/Users/eric/Desktop/andy/2b.cdp',
    #                      header=None, names=('cdp', 'x', 'y'), sep='\s+')
    # lon, lat = utils.unproject('15', 'N', np.array(data['x']), np.array(data['y']))
    if write_csv:
        with open(''.join([csv_save_file, '_latlong.csv']), 'w') as f:
            for site in data.site_names:
                f.write('{}, {:>10.5f}, {:>10.5f}, {:>5.5f}\n'.format(site,
                                                                      data.sites[site].locations['Long'],
                                                                      data.sites[site].locations['Lat'],
                                                                      data.sites[site].locations['elev']))

        with open(''.join([csv_save_file, '_UTM', str(UTM), '.csv']), 'w') as f:
            p = Proj(proj='utm', zone=UTM, ellps='WGS84')
            for site in data.site_names:
                x, y = p(data.sites[site].locations['Long'],
                         data.sites[site].locations['Lat'])
                f.write('{}, {:>10.1f}, {:>10.1f}, {:>5.1f}\n'.format(site,
                                                                      x,
                                                                      y,
                                                                      data.sites[site].locations['elev']))

    data.locations = data.get_locs(mode='latlong')
    kml = simplekml.Kml()
    for site in data.site_names:
    # for ii, (index, row) in enumerate(data.iterrows()):
        # if (ii % 10) == 0:
        lat, lon, elev = (data.sites[site].locations['Lat'],
                          data.sites[site].locations['Long'],
                          data.sites[site].locations['elev'])
            # kml.newpoint(name=str(row['cdp']), coords=[(lon[ii], lat[ii], 0)])
        kml.newpoint(name=site, coords=[(lon, lat, elev)])

    kml.save(kml_save_file)
