from collections import OrderedDict
# import harmonica as hm
# import pyproj
import numpy as np
# import matplotlib.pyplot as plt
# import pyMT.e_colours.colourmaps as cm
import pyMT.data_structures as DS
import pyMT.utils as utils
# from copy import deepcopy
from datetime import datetime
import pkg_resources
import os


def to_edi(site, out_file, info=None, header=None, mtsect=None, defs=None):
    lat_deg, lat_min, lat_sec = utils.dd_to_dms(site.locations['Lat'])
    long_deg, long_min, long_sec = utils.dd_to_dms(site.locations['Long'])
    default_header = OrderedDict([('ACQBY', '"eroots"'),
                                  ('FILEBY',   '"pyMT"'),
                                  ('DATAID', '{}'.format(site.name)),
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
    default_mtsect = OrderedDict([('SECTID', '{}'.format(site.name)),
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

    z_scale = 4 * np.pi / 10000

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
            freq = round(1 / p, 6)
            f.write('{:>14.4E}'.format(freq))
        f.write('\n\n')

        # Impedance info
        f.write('>ZROT //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>14.3f}'.format(site.azimuth))
        f.write('\n\n')

        f.write('>ZXXR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['ZXXR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZXXI  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(-1*site.data['ZXXI'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZXX.VAR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['ZXXR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYYR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['ZYYR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYYI  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(-1*site.data['ZYYI'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYY.VAR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['ZYYR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZXYR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['ZXYR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZXYI  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(-1*site.data['ZXYI'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZXY.VAR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['ZXYR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYXR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.data['ZYXR'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYXI  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(-1*site.data['ZYXI'][ii]/z_scale))
        f.write('\n\n')

        f.write('>ZYX.VAR  ROT=ZROT    //{}\n'.format(site.NP))
        for ii in range(site.NP):
            f.write('{:>18.7E}'.format(site.errors['ZYXR'][ii]/z_scale))
        f.write('\n\n')

        if 'TZXR' in site.components:
            # Tipper info
            f.write('>TROT.EXP //{}\n'.format(site.NP))
            for ii in range(site.NP):
                f.write('{:>14.3f}'.format(site.azimuth))
            f.write('\n\n')

            f.write('>TXR.EXP //{}\n'.format(site.NP))
            for ii in range(site.NP):
                f.write('{:>18.7E}'.format(site.data['TZXR'][ii]))
            f.write('\n\n')

            f.write('>TXI.EXP //{}\n'.format(site.NP))
            for ii in range(site.NP):
                f.write('{:>18.7E}'.format(-1*site.data['TZXI'][ii]))
            f.write('\n\n')

            f.write('>TYR.EXP //{}\n'.format(site.NP))
            for ii in range(site.NP):
                f.write('{:>18.7E}'.format(site.data['TZYR'][ii]))
            f.write('\n\n')

            f.write('>TYI.EXP //{}\n'.format(site.NP))
            for ii in range(site.NP):
                f.write('{:>18.7E}'.format(-1*site.data['TZYI'][ii]))
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

# path = 'E:/phd/NextCloud/data/Regions/Ciomadul/j2/originals/rotated/fixed/'
# path = 'E:/Work/sync/Regions/geraldton/edis/'
# list_path = 'E:/Work/sync/Regions/Ciomadul/j2/cio2021.lst'
# out_path = 'E:/Work/sync/Regions/Ciomadul/j2/2021_mtsect-fix/'
# list_path = 'E:/Work/sync/Regions/ATHA/j2/industry_AMT/S4-74H13-0014/j_format/allall.lst'
# out_path = 'E:/Work/sync/Regions/ATHA/j2/industry_AMT/S4-74H13-0014/rewritten_edis/'
# list_path = 'E:/Work/sync/Regions/ATHA/j2/industry_AMT/S3-74H13-0013/allall.lst'
# out_path = 'E:/Work/sync/Regions/ATHA/j2/industry_AMT/S3-74H13-0013/temp/'
# list_path = 'E:/Work/sync/Regions/undercover/j2/kuusamo_original/allall.lst'
# out_path = 'E:/Work/sync/Regions/undercover/j2/kuusamo_original/fix_header/'
list_path = r'E:/Work/sync/Regions/Reykjanes/edis/allall.lst'
out_path = r'E:/Work/sync/Regions/Reykjanes/edis/fix_header/'
# files = [x for x in os.listdir(path) if x.endswith('edi')]

data = DS.RawData(list_path)
for ii, site in enumerate(data.site_names):
    #######################
    # Add whatever extra logic in here you need to fix the EDIs
    # new_name = 'C2021_{}'.format(site[1:4])
    # new_name = 'C2021_{:03d}'.format(ii+1)
    # data.sites[site].name = 'S5_{}'.format(data.sites[site].name[3:])
    new_name = data.sites[site].name
    out_file = out_path + new_name + '.edi'
    # Replace some aspects of the header with the original info
    # header = {'ACQBY': 'GTK',
    #           'FILEBY': 'tf_mt',
              # 'COUNTRY': 'Finland'}#,
    #           'STATE': 'LX'}
    header = {}
    info = {}
    mtsect = {}
    defs = {}
    # data.sites[site].data['ZXYR'] *= -1
    # data.sites[site].data['ZXYI'] *= -1
    # data.sites[site].data['ZYXR'] *= -1
    # data.sites[site].data['ZYXI'] *= -1
    to_edi(data.sites[site], out_file, info=info, header=header, mtsect=mtsect, defs=defs)