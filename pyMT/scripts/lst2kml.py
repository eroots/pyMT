import simplekml
import pyMT.data_structures as WSDS
import pandas as pd
import numpy as np
import pyMT.utils as utils
from pyproj import Proj
import shapefile


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
             'swayze',
             'sudbury',
             'cobalt',
             'all',
             'bb')
lists = ('ATTBB.lst',
         'CHIBB.lst',
         'DRYBB.lst',
         'GERBB.lst',
         'LARBB.lst',
         'MALBB.lst',
         'MATBB.lst',
         'RRVBB.lst',
         'ROUBB.lst',
         'STUBB.lst',
         'SWZBB.lst',
         'SUDBB.lst',
         'COBBB.lst',
         'all.lst',
         'BB.lst')
# transects = ['all-legacy']
# lists = ['non-duplicates.lst']
# data_type = 'all_'
data_type = 'BB_'
# list_path = 'C:/Users/eric/phd/ownCloud/Metal Earth/Data/ConvertedEDIs/FinalEDIs/'
# list_path = 'F:/ownCloud/Metal Earth/Data/ConvertedEDIs/FinalEDIs/'
# list_path = 'F:/ownCloud/Metal Earth/Data/legacy_edi_export_all/'
list_path = 'F:/ownCloud/Metal Earth/Data/WinGLinkEDIs_final/'
csv_save_path = 'F:/ownCloud/Metal Earth/Data/MT-locations/CSVs/'
shp_save_path = 'F:/ownCloud/Metal Earth/Data/MT-locations/SHPs/'
kml_save_path = 'F:/ownCloud/Metal Earth/Data/MT-locations/KMLs/'
# csv_save_path = r'C:/Users/eric/phd/ownCloud/Metal Earth/Data/MT-locations/CSVs/'
write_kml = True
write_csv = True
write_shp = True
UTM = 16
for ii, lst in enumerate(lists):
    list_file = ''.join([list_path, lst])
    kml_save_file = ''.join([kml_save_path, data_type, transects[ii], '.kml'])
    csv_save_file = ''.join([csv_save_path, transects[ii]])
    shp_save_file = ''.join([shp_save_path, data_type, transects[ii], '.shp'])
    data = WSDS.RawData(listfile=list_file)
    # data = pd.read_table('C:/Users/eric/Desktop/andy/2b.cdp',
    #                      header=None, names=('cdp', 'x', 'y'), sep='\s+')
    # lon, lat = utils.unproject('15', 'N', np.array(data['x']), np.array(data['y']))
    if write_csv:
        with open(''.join([csv_save_file, data_type, '_latlong.csv']), 'w') as f:
            print('Writing CSV for {}'.format(transects[ii]))
            f.write('ID, Longitude, Latitude, Elevation\n')
            for site in data.site_names:
                f.write('{}, {:>10.5f}, {:>10.5f}, {:>5.5f}\n'.format(site,
                                                                      data.sites[site].locations['Long'],
                                                                      data.sites[site].locations['Lat'],
                                                                      data.sites[site].locations['elev']))

        with open(''.join([csv_save_file, data_type, '_UTM', str(UTM), '.csv']), 'w') as f:
            p = Proj(proj='utm', zone=UTM, ellps='WGS84')
            f.write('ID, Easting, Northing, Elevation\n')
            for site in data.site_names:
                x, y = p(data.sites[site].locations['Long'],
                         data.sites[site].locations['Lat'])
                f.write('{}, {:>10.1f}, {:>10.1f}, {:>5.1f}\n'.format(site,
                                                                      x,
                                                                      y,
                                                                      data.sites[site].locations['elev']))
    if write_kml:
        print('Writing KML for {}'.format(transects[ii]))
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

    if write_shp:
        print('Writing SHP for {}'.format(transects[ii]))
        if data.site_names:
            w = shapefile.Writer(shapefile.POINTM)
            w.field('Location')
            # w.field('Latitude', 'N')
            # w.field('Elevation', 'N')
            for site in data.site_names:
                lat, lon, elev = (data.sites[site].locations['Lat'],
                                  data.sites[site].locations['Long'],
                                  data.sites[site].locations['elev'])
                w.point(lon, lat, elev)
                w.record(site, 'Station')
            w.save(shp_save_file)
            # w.close()
