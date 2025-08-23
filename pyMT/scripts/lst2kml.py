import simplekml
import pyMT.data_structures as WSDS
import pandas as pd
import numpy as np
import pyMT.utils as utils
from pyproj import Proj
import shapefile
from pyMT.WSExceptions import WSFileError

# If you have multiple lists / lines that you want to keep seperate, add them all here. Otherwise, just rename the transect (but keep it in brackets as is)
# transects = ['litho_bb', 'litho_lmt', 'metal_earth', 'usarray']
transects = ['atha_mcarthir']
# Path to the list file
# list_path = 'E:/Work/sync/Regions/Norway/GoFEM/data/edi_interpolated/'
list_path = 'E:/Work/sync/Regions/ATHA/gofem/edi_interpolated/mcarthur/'
transects = ['mcarthur']
list_path = r'C:/Users/eroots/phd/Nextcloud/data/Regions/MSH/MSH-EDI_2022-23/MSH_EDI_trueN/impedances/'
transects = ['msh-T2']
# Path to the list file
# list_path = 'D:/Work/ATHA/Atha_21_processed_EDI/impedances/'
# dat_path = 'E:/phd/NextCloud/data/Regions/churchill/j2/mtpy/zero_azimuth/'
# list_path = 'E:/phd/Nextcloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/cullmantle_separated/'
# dat_path = 'E:/phd/Nextcloud/data/Regions/MetalEarth/wst/j2/mantle/fullrun/'
# Name of the list file(s) - these would go with your list of transects above
# lists = ['allsites.lst']
# lists = ['litho_bb.lst', 'litho_lmt.lst', 'metal_earth.lst', 'usarray.lst']
lists = ['allall.lst']
# Just a prefix for naming. Previously I had used it to separate AMT/BBMT/LMT data into separate files.
# data_type = 'BB_'
data_type = ''

# Set the paths you want things to be saved to
# shp_save_path = 'E:/phd/Nextcloud/data/ArcMap/MT-locations/wst_cullmantle/'
# kml_save_path = 'E:/phd/Nextcloud/data/ArcMap/MT-locations/KMLs/'
kml_save_path = 'C:/Users/eroots/phd/Nextcloud/data/ArcMap/MT-locations/KMLs/'
# csv_save_path = 'E:/phd/Nextcloud/data/ArcMap/MT-locations/KMLs/'
shp_save_path = 'E:/phd/NextCloud/data/ArcMap/MT-locations/stations/'
# kml_save_path = 'C:/Users/eroots/phd/Nextcloud/data/ArcMap/MT-locations/KMLs/'
csv_save_path = 'E:/phd/Nextcloud/data/ArcMap/MT-locations/KMLs/'

# Set to true or false depending on what you want written
write_kml = True
write_csv = False
write_shp = True
# Whatever your UTM zone is
UTM = 13

for ii, lst in enumerate(lists):
    try:
        list_file = ''.join([list_path, lst])
        kml_save_file = ''.join([kml_save_path, data_type, transects[ii], '.kml'])
        csv_save_file = ''.join([csv_save_path, transects[ii]])
        shp_save_file = ''.join([shp_save_path, data_type, transects[ii], '.shp'])
        data = WSDS.RawData(listfile=list_file)
        if write_csv:
            with open(''.join([csv_save_file, data_type, '_latlong.csv']), 'w') as f:
                print('Writing CSV for {}'.format(transects[ii]))
                f.write('ID, Longitude, Latitude, Elevation/n')
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
                # w = shapefile.Writer(shp_save_file)
                with shapefile.Writer(target=shp_save_file) as w:
                    w.field('X', 'F', 10, 5)
                    w.field('Y', 'F', 10, 5)
                    # w.field('Z', 'F', 10, 5)
                    w.field('Label')
                    for site in data.site_names:
                        lat, lon, elev = (data.sites[site].locations['Lat'],
                                          data.sites[site].locations['Long'],
                                          data.sites[site].locations['elev'])
                        w.point(lon, lat)
                        w.record(lon, lat, site)
                # w.save(shp_save_file)
                # w.close()
    except WSFileError as e:
        print('List {} not found'.format(list_file))
