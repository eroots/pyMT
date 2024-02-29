# Converts ZTEM data from Geosoft .grd format to EDI files.
# Note that the .grd data may be interpolated from the actual flight lines, so choose the downsampling rate accordingly.

# Set your parameters here

# Specify the folder containing the grid files
grid_path = 'E:/Work/ztem_test/Geosoft_grids/'
# Files should be named as <grid_tag>_<component>_<frequency>.grid
# E.g., GL210135_XQD_030Hz.grd is the X-Quadrature component at 30 Hz
grid_tag = 'GL210135'
# Path to where you want the files to be output
out_path = 'E:/Work/ztem_test/new_edis/'
# UTM zone so that the lat/longs can be calculated from the grid coordinates
utm_zone = 10
# List of frequencies available
freqs = [30, 45, 90, 180, 360, 720][::-1]
# Downsampling rate (grid will be coarsened by this ratio)
downsample_rate = 2
# Flat error floor to be applied in the EDI file.
flat_error = 0.03

# Don't have to change anything after this
from collections import OrderedDict
import harmonica as hm
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import pyMT.e_colours.colourmaps as cm
import pyMT.data_structures as DS
import pyMT.utils as utils
from copy import deepcopy
from datetime import datetime
# import pkg_resources


def to_edi(site, out_file, info=None, header=None, mtsect=None, defs=None):
    lat_deg, lat_min, lat_sec = utils.dd_to_dms(site.locations['Lat'])
    long_deg, long_min, long_sec = utils.dd_to_dms(site.locations['Long'])
    default_header = OrderedDict([('ACQBY', '"eroots"'),
                                  ('FILEBY',   '"pyMT"'),
                                  ('FILEDATE', datetime.today().strftime('%m/%d/%y')),
                                  ('LAT', '{:d}:{:d}:{:4.2f}'.format(int(lat_deg), int(lat_min), lat_sec)),
                                  ('LONG', '{:d}:{:d}:{:4.2f}'.format(int(long_deg), int(long_min), long_sec)),
                                  ('ELEV', 0),
                                  ('STDVERS', '"SEG 1.0"'),
                                  ('PROGVERS', '"pyMT {}"'.format(pkg_resources.get_distribution('pyMT').version)),
                                  ('COUNTRY', 'CANADA'),
                                  ('EMPTY', 1.0e+32)])
    default_info = OrderedDict([('MAXINFO', 999),
                                ('SURVEY ID', '""')])

    default_defs = OrderedDict([('MAXCHAN', 1),
                                ('MAXRUN', 999),
                                ('MAXMEAS', 9999),
                                ('UNITS', 'M'),
                                ('REFTYPE', 'CART'),
                                ('REFLAT', '{:d}:{:d}:{:4.2f}'.format(int(lat_deg), int(lat_min), lat_sec)),
                                ('REFLONG', '{:d}:{:d}:{:4.2f}'.format(int(long_deg), int(long_min), long_sec))])
    default_mtsect = OrderedDict([('SECTID', '""'),
                                  ('NFREQ', site.NP),
                                  ('HX', '1.01'),
                                  ('HY', '2.01'),
                                  ('HZ', '3.01')])
    if info:
        for key, val in info.items():
            default_info.update({key: val})
    info = default_info
    if header:
        for key, val in header.items():
            default_header.update({key: val})
    header = default_header
    if mtsect:
      for key, val in mtsect.items():
            default_mtsect.update({key: val})
    mtsect = default_mtsect
    if defs:
      for key, val in defs.items():
            default_defs.update({key: val})
    defs = default_defs

    # Write the file
    with open(out_file, 'w') as f:
        f.write('>HEAD\n')
        for key, val in header.items():
            f.write('{}={}\n'.format(key, val))
        f.write('\n')
        
        f.write('>INFO\n')
        for key, val in info.items():
            f.write('{}={}\n'.format(key, val))
        f.write('\n')

        f.write('>=DEFINEMEAS\n')
        for key, val in defs.items():
            f.write('{}={}\n'.format(key, val))
        f.write('\n')
        
        f.write('>=MTSECT\n')
        for key, val in mtsect.items():
            f.write('{}={}\n'.format(key, val))
        f.write('\n')

        f.write('>FREQ //{}\n'.format(site.NP))
        for p in site.periods:
            freq = round(1 / p, 3)
            f.write('{:>14.4E}'.format(freq))
        f.write('\n\n')

        f.write('>TROT //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>14.3f}'.format(site.azimuth))
        f.write('\n\n')

        f.write('>TXR.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['TZXR'][ii]))
        f.write('\n\n')

        f.write('>TXI.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['TZXI'][ii]))
        f.write('\n\n')

        f.write('>TYR.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['TZYR'][ii]))
        f.write('\n\n')

        f.write('>TYI.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['TZYI'][ii]))
        f.write('\n\n')

        f.write('>TXVAR.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['TZXR'][ii]))
        f.write('\n\n')
        f.write('>TYVAR.EXP //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['TZYR'][ii]))
        f.write('\n\n')

        f.write('>END')


def main():


    periods = [round(1/x, 15) for x in freqs]
    dummy_data = {'TZXR': np.zeros(len(freqs)),
                  'TZXI': np.zeros(len(freqs)),
                  'TZYR': np.zeros(len(freqs)),
                  'TZYI': np.zeros(len(freqs))}
    dummy_errors = {'TZXR': flat_error + np.zeros(len(freqs)),
                    'TZXI': flat_error + np.zeros(len(freqs)),
                    'TZYR': flat_error + np.zeros(len(freqs)),
                    'TZYI': flat_error + np.zeros(len(freqs))}
    source_crs = 'epsg:326{:02d}'.format(utm_zone)
    target_crs = 'epsg:4326'
    projection = pyproj.Transformer.from_crs(source_crs, target_crs)
    convert = {'XIP': 'TZXR', 'YIP': 'TZYR', 'XQD': 'TZXI', 'YQD': 'TZYI'}
    data = {'XIP': [], 'YIP': [], 'XQD': [], 'YQD': []}
    orig_data = {'XIP': [], 'YIP': [], 'XQD': [], 'YQD': []}


    for ip, freq in enumerate(freqs):
        for ic, component in enumerate(['XIP', 'YIP', 'XQD', 'YQD']):
            grid_file = '{}{}_{}_{:03d}Hz.grd'.format(grid_path, grid_tag, component, freq)
            grd = hm.load_oasis_montaj_grid(grid_file)
            ds_grd = grd.coarsen(easting=downsample_rate,
                                 northing=downsample_rate,
                                 boundary='trim').mean()
            X, Y = np.meshgrid(ds_grd.easting, ds_grd.northing)
            idx = np.isnan(ds_grd)
            new_lat, new_lon = projection.transform(X, Y)
            if component == 'XIP' and ip == 0:
                all_data = np.zeros((new_lat.size, 4, len(freqs)))
                old_lat, old_lon = new_lat, new_lon
                site_names = [str(ii) for ii in range(len(X.flatten()))]
                site_lats = new_lat.flatten()
                site_lons = new_lon.flatten()
            else:
                if not np.all(np.isclose(new_lat, old_lat)):
                    print('Latitudes differ between grids')
            data[component] = ds_grd.data
            orig_data[component] = grd.data
            
            all_data[:, ic, ip] = data[component].flatten()

    site_data = {name: [] for name in site_names}
    rm_sites = []
    for jj, (x, y) in enumerate(zip(site_lats, site_lons)):
        site_name = site_names[jj]
        
        print('Initializing site {}'.format(jj))
        if np.any(np.isnan(all_data[jj, :, :])):
            print('Station {} has NaNs. Skipping...'.format(jj))
            rm_sites.append(site_name)
            continue
        # Report says time dependence is -iwt, so should be flipped if writing to EDI?
        # Apparently not? This setup seems OK, which means the time dependence is already correct and they must have reals reversed?
        # 
        # Is there any rotation in the data?
        d = {'TZYR': -1 * all_data[jj, 0, :],
             'TZXR': -1 * all_data[jj, 1, :],
             'TZYI': all_data[jj, 2, :],
             'TZXI': all_data[jj, 3, :]}
        site_data.update({site_name: DS.Site(name=site_name,
                                             periods=periods,
                                             data=d,
                                             errors=dummy_errors,
                                             errmap=None,
                                             locations={'Lat': x, 'Long': y},
                                             azimuth=0,
                                             flags=None)})

    # Remove any data from the coarsened grid that has NaNs
    for rm in rm_sites:
            site_names.remove(rm)
    mt_data = DS.Data()
    mt_data.site_names = site_names
    mt_data.sites = site_data
    for site in mt_data.site_names:
        out_file = out_path + site + '.edi'
        to_edi(mt_data.sites[site], out_file, info=None, header=None, mtsect=None, defs=None)

if __name__ == '__main__':
    main()

# plt.figure()
# for ii, comp in enumerate(data.keys()):
#     plt.subplot(2, 2, ii + 1)
#     plt.pcolor(new_lon, new_lat, data[comp], cmap=cm.get_cmap('turbo', N=32, invert=True))
# plt.figure()
# for ii, comp in enumerate(data.keys()):
#     plt.subplot(2, 2, ii + 1)
#     plt.pcolor(orig_data[comp], cmap=cm.get_cmap('turbo', N=32, invert=True))

