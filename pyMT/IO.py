from pyMT.WSExceptions import WSFileError
import pyMT.utils as utils
import numpy as np
import os
import copy
import re


if os.name is 'nt':
    PATH_CONNECTOR = '\\'
else:
    PATH_CONNECTOR = '/'


def verify_input(message, expected, default=None):
        while True:
            ret = input(' '.join([message, '[Default: {}] > '.format(default)]))
            if ret == '' and default is not None:
                ret = default
            if expected == 'read':
                if utils.check_file(ret):
                    return ret
                else:
                    print('File not found. Try again.')
            elif expected == 'write':
                if ret:
                    if not utils.check_file(ret):
                        return ret
                    else:
                        resp = verify_input('File exists, overwrite?', default='y', expected='yn')
                        if resp == 'y':
                            return ret
                        else:
                            return False
                else:
                    print('Output file name required!')
            elif expected == 'numtuple':
                try:
                    ret = [float(x) for x in re.split(', | ', ret)]
                    return ret
                except ValueError:
                    print('Could not convert {} to tuple'.format(ret))
            elif isinstance(expected, str):
                try:
                    if ret.lower() not in expected:
                        print('That is not an option. Try again.')
                    else:
                        return ret.lower()
                except AttributeError:  # Catch case where string is a digit
                    if str(ret) not in expected:
                        print('That is not an option. Try again.')
                    else:
                        return ret
            else:
                try:
                    return expected(ret)
                except ValueError:
                    print('Format error. Try again')


def get_components(invType=None, NR=None):
    """Summary

    Args:
        invType (TYPE): Description
        NR (TYPE): Description

    Returns:
        TYPE: Description
    """
    possible = ('ZXXR', 'ZXXI',
                'ZXYR', 'ZXYI',
                'ZYXR', 'ZYXI',
                'ZYYR', 'ZYYI',
                'TZXR', 'TZXI',
                'TZYR', 'TZYI')
    # print(invType)
    if not NR:
        NR = 0
    if (NR == 8 and not invType) or (invType == 1):
        comps = [possible[0:8], tuple(range(8))]
    elif (NR == 12 and not invType) or (invType == 5):
        comps = [possible, tuple(range(12))]
    elif (NR == 4 and not invType) or (invType == 2):
        comps = [possible[2:6], tuple(range(4))]
    elif invType == 3:
        comps = [possible[8:], tuple(range(4))]
    elif invType == 4:
        # comps = [possible[2, 3, 4, 5, 8, 9, 10, 11],
        comps = [possible[2:6] + possible[8:12],
                 tuple(range(8))]
        # tuple(range(8))]
    elif invType == 5:
        comps = possible
    return comps


def read_model(modelfile=''):
    if not modelfile:
        return None

    with open(modelfile, 'r') as f:
        mod = {'xCS': [], 'yCS': [], 'zCS': [], 'vals': []}
        # while True:
        header = next(f)
        header = next(f)
        # if header[0] != '#':
        # break
        NX, NY, NZ, *MODTYPE = [h for h in header.split()]
        NX, NY, NZ = int(NX), int(NY), int(NZ)
        loge_flag = False
        if len(MODTYPE) == 1:
            MODTYPE = int(MODTYPE[0])
        else:
            if MODTYPE[1] == 'LOGE':
                loge_flag = True
            MODTYPE = int(MODTYPE[0])
        lines = f.readlines()
        lines = [x.split() for x in lines]
        lines = [item for sublist in lines for item in sublist]
        lines = [float(x) for x in lines]
        mod['xCS'] = lines[:NX]
        mod['yCS'] = lines[NX: NX + NY]
        mod['zCS'] = lines[NX + NY: NX + NY + NZ]
        if MODTYPE == 1:
            mod['vals'] = np.ones([NX, NY, NZ]) * lines[NX + NY + NZ]
        else:
            # vals = lines[NX + NY + NZ: ]
            mod['vals'] = np.flipud(np.reshape(np.array(lines[NX + NY + NZ: NX + NY + NZ + NX * NY * NZ]),
                                               [NX, NY, NZ], order='F'))
        if loge_flag:
            mod['vals'] = np.exp(mod['vals'])
        return mod


def read_raw_data(site_names, datpath=''):
    """Summary

    Args:
        site_names (TYPE): Description
        datpath (str, optional): Description

    Returns:
        TYPE: Description
    """
    RAW_COMPONENTS = ('ZXX', 'ZXY',
                      'ZYX', 'ZYY',
                      'TZX', 'TZY')

    def read_dat(file, long_origin=999):

        try:
            with open(file, 'r') as f:
                siteData_dict = {}
                siteError_dict = {}
                siteLoc_dict = {'azi': [], 'Lat': [], 'Long': [], 'elev': []}
                while True:
                    L = next(f)  # Header, unused
                    if L[0] != '#':
                        break
                while True:
                    if L[0] == '>':
                        if 'azimuth' in L.lower():
                            siteLoc_dict.update({'azi': float(L.split('=')[1].rstrip('\n'))})
                        elif 'latitude' in L.lower():
                            Lat = float(L.split('=')[1].rstrip('\n'))
                        elif 'longitude' in L.lower():
                            Long = float(L.split('=')[1].rstrip('\n'))
                        elif 'elevation' in L.lower():
                            siteLoc_dict.update({'elev': float(L.split('=')[1].rstrip('\n'))})
                        L = next(f)
                    else:
                        break
                Y, X, long_origin = utils.geo2utm(Lat, Long, long_origin=long_origin)
                siteLoc_dict.update({'X': X, 'Y': Y})
                siteLoc_dict.update({'Lat': Lat, 'Long': Long})
                next(f)  # Another unused line (site name and a 0)
                lines = f.read().splitlines()
                comps = [(ii, comp[:3]) for ii, comp in enumerate(lines)
                         if comp[:3] in RAW_COMPONENTS]
                ns = [int(lines[int(ii[0]) + 1]) for ii in comps]
                ns_cache = []
                for ii, comp in comps:
                    ns = int(lines[ii + 1])
                    ns_cache.append(ns)
                    data = [(row.split()) for row in lines[ii + 2: ii + ns + 2]]
                    data = np.array([row for row in data])
                    data = data.astype(np.float)
                    periods = data[:, 0]
                    dataReal = data[:, 1]
                    dataImag = data[:, 2]
                    dataErr = data[:, 3]
                    siteData_dict.update({''.join([comp, 'R']): dataReal})
                    siteData_dict.update({''.join([comp, 'I']): -1 * dataImag})
                    siteError_dict.update({''.join([comp, 'R']): dataErr})
                    siteError_dict.update({''.join([comp, 'I']): dataErr})
                periods[periods < 0] = -1 / periods[periods < 0]
        except FileNotFoundError as e:
            raise(WSFileError(ID='fnf', offender=file))
        try:
            assert len(set(ns_cache)) == 1
        except AssertionError:
            msg = 'Number of periods in {} is not equal for all components'.format(file)
            print('Fatal error in pyMT.IO.read_dat')
            raise(WSFileError(id='int', offender=file, extra=msg))
        site_dict = {'data': siteData_dict,
                     'errors': siteError_dict,
                     'locations': siteLoc_dict,
                     'periods': periods,
                     'azimuth': siteLoc_dict['azi']
                     }
        return site_dict, long_origin

        # For now I'm only concerned with reading impedances and tipper
        # I make some assumptions, that may have to be changed later:
        #   Rotation angles are all the same
    def read_edi(file, long_origin=999):
        # For now I'm only concerned with reading impedances and tipper
        # I make some assumptions, that may have to be changed later:
        #   Rotation angles are all the same
        #   Negatives in Lat/Long are ignored
        def extract_blocks(lines):
            blocks = {'HEAD': [],
                      'ZROT': [],
                      'ZXXR': [],
                      'ZXXI': [],
                      'ZXYR': [],
                      'ZXYI': [],
                      'ZYXR': [],
                      'ZYXI': [],
                      'ZYYR': [],
                      'ZYYI': [],
                      'ZXX.VAR': [],
                      'ZXY.VAR': [],
                      'ZYX.VAR': [],
                      'ZYY.VAR': [],
                      'TXR.EXP': [],
                      'TXI.EXP': [],
                      'TXVAR.EXP': [],
                      'TYR.EXP': [],
                      'TYI.EXP': [],
                      'TYVAR.EXP': [],
                      'FREQ': [],
                      'INFO': [],
                      '=DEFINEMAS': []}
            in_block = 0
            ii = 0
            while ii < len(lines):
                line = lines[ii]
                if (not in_block and any(''.join(['>', key]) in line for key in blocks.keys())):
                    block_start = ii
                    in_block = 1
                    block_name = [key for key in blocks.keys()
                                  if ''.join(['>', key]) in line][0]
                    # print(block_name + str(block_start))
                elif in_block and line[0] == '>':
                    in_block = 0
                    blocks.update({block_name: lines[block_start:ii]})
                    # print(block_name + str(ii))
                    ii -= 2
                ii += 1
            return blocks

        def read_header(header):
            for line in header:
                if 'LAT' in line:
                    lat = (utils.dms2dd(line.split('=')[1].strip()))
                if 'LONG' in line:
                    lon = (utils.dms2dd(line.split('=')[1].strip()))
                if 'ELEV' in line:
                    elev = float(line.split('=')[1].strip())
            return lat, lon, elev

        def read_data_block(block):
            # print(block[0])
            num_freqs = float(block[0].split('//')[1])
            data = []
            for line in block[1:]:
                for f in line.split():
                    data.append(float(f.strip()))
            if len(data) != num_freqs:
                print('Number of frequencies does not match the given number')
                print('Proceeding anyways...')
            return np.array(data)

        def extract_tensor_info(blocks):
            scaling_factor = 4 * np.pi / 10000
            for key in blocks.keys():
                # print(key)
                if (key[0] == 'Z' or key[0] == 'T') and key != 'ZROT':
                    if blocks[key]:
                        data_block = read_data_block(blocks[key])
                        if key[0] == 'Z':
                            new_key = key[:4]
                        else:
                            new_key = ''.join(['TZ', key[1:3]])
                        if 'VAR' in key:
                            errors.update({new_key[:-1] + 'R': data_block * scaling_factor})
                            errors.update({new_key[:-1] + 'I': data_block * scaling_factor})
                        else:
                            data.update({new_key: data_block * scaling_factor})
                elif key == 'ZROT':
                    data_block = read_data_block(blocks[key])
                    if not np.all(data_block == data_block[1]):
                        print('Not all rotations are the same. This is not supported yet...')
                        azi = data_block[0]
                    else:
                        azi = data_block[0]
            # EDI format has ZxyR, ZxyI positive; ZyxR, ZyxI negative. This needs to be changed
            data['ZXYI'] *= -1
            data['ZYXI'] *= -1
            data['ZXXI'] *= -1
            data['ZYYI'] *= -1
            return data, errors, azi

        data = {}
        errors = {}
        with open(file, 'r') as f:
            # Need to also read in INFO and DEFINEMEAS blocks to confirm that the location
            # info is consistent
            lines = f.readlines()
            blocks = extract_blocks(lines)
            Lat, Long, elev = read_header(blocks['HEAD'])
            frequencies = read_data_block(blocks['FREQ'])
            periods = utils.truncate(1 / frequencies)
            data, errors, azi = extract_tensor_info(blocks)
            Y, X, long_origin = utils.geo2utm(Lat, Long, long_origin=long_origin)
            location_dict = {'X': X, 'Y': Y, 'Lat': Lat, 'Long': Long, 'elev': elev}
            site_dict = {'data': data,
                         'errors': errors,
                         'locations': location_dict,
                         'periods': periods,
                         'azimuth': azi
                         }

            # return blocks, lines, freqs
            return site_dict, long_origin

    siteData = {}
    long_origin = 999
    if os.name is 'nt':
        connector = '\\'
    else:
        connector = '/'
    if datpath:
        path = datpath + connector
    else:
        path = './'
    all_dats = [file for file in os.listdir(path) if file.endswith('.dat')]
    # all_edi = [file for file in os.listdir(path) if file.endswith('.edi')]
    for site in site_names:
        # Look for J-format files first
        if ''.join([site, '.dat']) in all_dats:
            file = ''.join([path, site, '.dat'])
            try:
                site_dict, long_origin = read_dat(file, long_origin)
            except FileNotFoundError:
                print('{} not found. Continuing without it.'.format(file))
        else:
            file = ''.join([path, site, '.edi'])
            try:
                site_dict, long_origin = read_edi(file, long_origin)
            except FileNotFoundError:
                print('{} not found. Continuing without it.'.format(file))
        siteData.update({site: site_dict})
    return siteData


def read_sites(listfile):
    """Summary

    Args:
        listfile (TYPE): Description

    Returns:
        TYPE: Description
    """
    try:
        with open(listfile, 'r') as f:
            ns = int(next(f))
            site_names = list(filter(None, f.read().split('\n')))
            site_names = [name.replace('.dat', '') for name in site_names]
            if ns != len(site_names):
                raise(WSFileError(ID='int', offender=listfile,
                                  extra='# Sites does not match length of list.'))
        return site_names
    except FileNotFoundError as e:
        raise(WSFileError('fnf', offender=listfile))


def read_startup(file=None):
    s_dict = {}
    if not file:
        file = 'startup'
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'INVERSION_TYPE' in line.upper():
                s_dict.update({'inv_type': int(line.split()[1])})
            elif 'MIN_ERROR_Z' in line.upper():
                s_dict.update({'errFloorZ': float(line.split()[1])})
            elif 'MIN_ERROR_T' in line.upper():
                s_dict.update({'errFloorT': float(line.split()[1])})
            elif 'DATA_FILE' in line.upper():
                s_dict.update({'datafile': line.split()[1]})
    return s_dict


def read_data(datafile='', site_names='', file_format='WSINV3DMT', invType=None):
    """Summary

    Args:
        datafile (str, optional): Description
        site_names (str, optional): Description
        filetype (str, optional): Description
        invType (None, optional): Description

    Returns:
        TYPE: Description
    """
    def read_ws_data(datafile='', site_names='', invType=None):
        try:
            with open(datafile, 'r') as f:
                NS, NP, *NR = [round(float(h), 1) for h in next(f).split()]
                if len(NR) == 1:
                    azi = 0  # If azi isn't specified, it's assumed to be 0
                else:
                    azi = NR[1]
                NR = int(NR[0])
                NS = int(NS)
                NP = int(NP)
                if not site_names:
                    site_names = [str(x) for x in list(range(0, NS))]
                if NS != len(site_names):
                    raise(WSFileError(ID='int', offender=datafile,
                                      extra='Number of sites in data file not equal to list file'))
                # Components is a pair of (compType, Nth item to grab)

                if os.path.split(datafile)[0]:
                    startup_file = PATH_CONNECTOR.join([os.path.split(datafile)[0], 'startup'])
                else:
                    startup_file = 'startup'
                if utils.check_file(startup_file):
                    startup = read_startup(startup_file)
                    invType = startup.get('inv_type', None)
                else:
                    startup = {}
                components = get_components(invType, NR)
                # next(f)  # skip the next line
                lines = f.readlines()
                for ii, l in enumerate(lines):
                    if '#' in l:
                        lines.pop(ii)
                sl = [ii for ii, l in enumerate(lines) if 'Station' in l]
                data = [line.split() for line in lines[sl[0] + 1: sl[1]]]
                xlocs = utils.flatten_list(data)
                spoints = [(ii, p) for ii, p in enumerate(lines)
                           if 'Period' in p]
                data = [line.split() for line in lines[sl[1] + 1: spoints[0][0]]]
                ylocs = utils.flatten_list(data)
                linenums = [ii[0] + 1 for ii in spoints]
                linenums.append(len(lines) + 1)
                comps = [comp[0] for comp in (zip(components[0], components[1]))]
                siteData = {site: {comp: [] for comp in comps}
                            for site in site_names}
                siteError = {site: {comp: [] for comp in comps}
                             for site in site_names}
                siteErrMap = {site: {comp: [] for comp in comps}
                              for site in site_names}
                # If there is a rounding error here, np.unique doesn't properly reduce the
                # periods to the right set. We know there are NP periods, just take those straight away.
                periods = np.unique(np.array([float(p[1].split()[1]) for p in spoints[:NP]]))
                for ii in range(len(linenums) - 1):
                    header = lines[linenums[ii] - 1]
                    data = [line.split() for line in
                            lines[linenums[ii]: linenums[ii + 1] - 1]]
                    data = utils.flatten_list(data)
                    # This runs, but it remains to be seen whether or not it is correct.
                    for jj, site in enumerate(site_names):
                        for kk in zip(components[0], components[1]):
                            ind = (NR * jj) + (kk[1])
                            comp = kk[0]
                            if 'DATA' in header:
                                siteData[site][comp].append(data[ind])
                            elif 'ERROR' in header:
                                siteError[site][comp].append(data[ind])
                            elif 'ERMAP' in header:
                                siteErrMap[site][comp].append(data[ind])
                for site in site_names:
                    for kk in zip(components[0], components[1]):
                        comp = kk[0]
                        vals = siteData[site][comp]
                        siteData[site][comp] = np.array(vals)
                        vals = siteError[site][comp]
                        siteError[site][comp] = np.array(vals)
                        vals = siteErrMap[site][comp]
                        siteErrMap[site][comp] = np.array(vals)
                sites = {}
                siteLocs = {}
                for ii, site in enumerate(site_names):
                    siteLocs = {'X': xlocs[ii], 'Y': ylocs[ii]}
                    sites.update({site: {
                                  'data': siteData[site],
                                  'errors': siteError[site],
                                  'errmap': siteErrMap[site],
                                  'periods': periods,
                                  'locations': siteLocs,
                                  'azimuth': azi,
                                  'errFloorZ': startup.get('errFloorZ', None),
                                  'errFloorT': startup.get('errFloorT', None)}
                                  })
        except FileNotFoundError as e:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        other_info = {'inversion_type': invType,
                      'site_names': site_names,
                      'UTM_zone': 'Undefined',
                      'origin': (0, 0)}
        return sites, other_info

    def read_modem_data(datafile='', site_names='', invType=None):
        #  Will only ready Impedance and TF data so far, not rho/phase
        # print('Inside read_modem_data')
        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        marker = -10
        block_start = []
        periods = []
        sites = {}
        site_data = {}
        site_error = {}
        # site_errmap = {}
        site_locations = {}
        inv_type = 0
        new_site_names = []
        if site_names:
            site_data = {site: {} for site in site_names}
            site_error = {site: {} for site in site_names}
            site_locations = {site: {'X': [], 'Y': [], 'elev': []} for site in site_names}
        for ii, line in enumerate(lines):
            if '#' in line and marker + 1 != ii:
                marker = ii
                block_start.append(ii)
        block_start.append(len(lines))
        for ii, line_number in enumerate(block_start[:-1]):
            header = lines[line_number + 1]
            if 'UTM Zone:' in header:
                UTM_zone = header.split(':')[1]
            else:
                UTM_zone = 'Undefined'
            data_type = lines[line_number + 2].split('>')[1].strip()
            azimuth = float(lines[line_number + 5].split('>')[1].strip())
            o_x, *o_y = [float(x) for x in lines[line_number + 6].split('>')[1].strip().split()]
            o_y = o_y[0]
            NP, NS = [int(x) for x in lines[line_number + 7].split('>')[1].strip().split()]
            if data_type == 'Full_Impedance':
                if inv_type == 3:
                    inv_type = 5
                else:
                    inv_type = 1
            #     components = ('ZXX', 'ZYY', 'ZXY', 'ZYR')
            elif data_type == 'Full_Vertical_Components':
                if inv_type == 1:
                    inv_type = 5
                elif inv_type == 2:
                    inv_type = 4
                else:
                    inv_type = 3
            #     components = ('TX', 'TY')
            elif data_type == 'Off_Diagonal_Impedance':
                if inv_type == 3:
                    inv_type = 4
                else:
                    inv_type = 2
            #     # components = ('ZXY', 'ZYX')
            for line_string in lines[block_start[ii] + 8: block_start[ii + 1]]:
                line = line_string.split()
                periods.append(float(line[0]))
                site_name = line[1]
                X, Y, Z = [float(x) for x in line[4:7]]
                if site_name not in site_data.keys():
                    site_data.update({site_name: {}})
                    site_error.update({site_name: {}})
                    new_site_names.append(site_name)
                    site_locations.update({site_name: {'X': [],
                                                       'Y': [],
                                                       'elev': []}})
                site_locations[site_name]['X'] = X
                site_locations[site_name]['Y'] = Y
                site_locations[site_name]['elev'] = Z

                component = line[7]
                real, imag = [float(x) for x in line[8:10]]
                error = float(line[10])
                if component == 'TX' or component == 'TY':
                    component = component[0] + 'Z' + component[1]
                if component + 'R' not in site_data[site_name].keys():
                    site_data[site_name].update({component + 'R': []})
                if component + 'I' not in site_data[site_name].keys():
                    site_data[site_name].update({component + 'I': []})
                if component + 'R' not in site_error[site_name].keys():
                    site_error[site_name].update({component + 'R': []})
                if component + 'I' not in site_error[site_name].keys():
                    site_error[site_name].update({component + 'I': []})
                # if component not in site_errmap.keys():
                #     site_errmap[site_name].update({component + 'R': []})
                #     site_errmap[site_name].update({component + 'I': []})
                site_data[site_name][component + 'R'].append(real)
                site_data[site_name][component + 'I'].append(imag)
                site_error[site_name][component + 'R'].append(error)
                site_error[site_name][component + 'I'].append(error)
                # print(site_data[site_lookup[code]][component + 'R'])
        periods = np.unique(np.array(periods))
        # print(site_data[site_names[0]])
                # site_errmap[site][component] = np.array(site_errmap[site][component])
        # print(site_data[site][component])
        if not site_names:
            site_names = new_site_names
        elif site_names != new_site_names:
            print('Site names specified in list file do not match those in {}\n'.format(datafile))
            print('Proceeding with names set in list file.\n')
            # site_names = new_site_names
        for site in new_site_names:
            for component in site_data[site].keys():
                vals = site_data[site][component]
                site_data[site][component] = np.array(vals)
                vals = site_error[site][component]
                site_error[site][component] = np.array(vals)
        for ii, site in enumerate(site_names):
            sites.update({site: {
                          'data': site_data[new_site_names[ii]],
                          'errors': site_error[new_site_names[ii]],
                          'periods': periods,
                          'locations': site_locations[new_site_names[ii]],
                          'azimuth': azimuth,
                          'errFloorZ': 0,
                          'errFloorT': 0}
                          })
        other_info = {'inversion_type': inv_type,
                      'site_names': site_names,
                      'origin': (o_x, o_y),
                      'UTM_zone': UTM_zone}
        return sites, other_info

    def read_mare2dem_data(datafile='', site_names='', invType=None):
        data_type_lookup = {101: 'RhoZXX',
                            102: 'PhsZXX',
                            103: 'RhoZXY',
                            104: 'PhsZXY',
                            105: 'RhoZYX',
                            106: 'PhsZYX',
                            107: 'RhoZYY',
                            108: 'PhsZYY',
                            111: 'ZXXR',
                            112: 'ZXXI',
                            113: 'ZXYR',
                            114: 'ZXYI',
                            115: 'ZYXR',
                            116: 'ZYXI',
                            117: 'ZYYR',
                            118: 'ZYYI',
                            121: 'log10RhoZXX',
                            123: 'log10RhoZXY',
                            125: 'log10RhoZYX',
                            127: 'log10RhoZYY',
                            133: 'TZYR',
                            134: 'TZYI'}
        #  Assumes you're only working with MT data
        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        #  Remove comment lines
        lines = [line for line in lines if not line.startswith('%') and not line.startswith('!')]
        em_format = lines[0].split(':')[1].strip()
        if 'emresp' in em_format.lower():
            em_format = 'Response'
        else:
            em_format = 'Data'
        #  It's not entirely clear how strike is used in the program
        UTM_zone, o_x, o_y, strike = re.split(r'\s{2,}', lines[1].split(':')[1].strip())
        o_x, o_y, strike = [float(x) for x in [o_x, o_y, strike]]
        #  Will have to look into the code to see what is done, but I think this makes sense
        azimuth = strike + 90
        NP = int(lines[2].split(':')[1].strip())
        frequencies = []
        for ii in range(NP):
            frequencies.append(float(lines[3 + ii]))
        periods = [1 / f for f in frequencies]
        periods = np.array(periods)
        lines = lines[3 + NP:]
        NS = int(lines[0].split(':')[1].strip())
        # value_ids = [value for value in lines[1].split()[1:]]
        value_ids = ['X', 'Y', 'Z', 'Theta', 'Alpha', 'Beta', 'Length', 'SolveStatic', 'Name']
        site_dict = {}
        site_names = []
        site_aliases = {x: [] for x in range(NS)}
        for ii, line in enumerate(lines[1:NS + 1]):
            site = {value_ids[ii]: value for ii, value in enumerate(line.split())}
            site_aliases.update({ii + 1: site['Name']})
            site_names.append(site['Name'])
            site_dict.update({site['Name']: {'locations': {'X': float(site['X']),
                                                           'Y': float(site['Y']),
                                                           'elev': float(site['Z'])}}})
            site_dict[site['Name']].update({'data': {},
                                            'errors': {}})
            site_dict[site['Name']].update({'solve_static': site['SolveStatic']})
            site_dict[site['Name']].update({'azimuth': site['Theta']})
        # num_data = int(lines[NS + 4].split(':')[1])
        # value_ids = [value for value in lines[NS + 3].split()[1:]]
        if em_format == 'Data':
            value_ids = ['Type', 'Freq#', 'Tx#', 'Rx#', 'Data', 'StdError']
        else:
            value_ids = ['Type', 'Freq#', 'Tx#', 'Rx#', 'Data', 'StdError', 'Response', 'Residual']
        for line in lines[NS + 2:]:
            site = {value_ids[ii]: float(value) for ii, value in enumerate(line.split())}
            site_name = site_aliases[int(site['Rx#'])]
            data_type = data_type_lookup[int(site['Type'])]
            if data_type_lookup[site['Type']] not in site_dict[site_name]['data'].keys():
                site_dict[site_name]['data'].update({data_type: [0 for x in range(NP)]})
                site_dict[site_name]['errors'].update({data_type: [0 for x in range(NP)]})
                # site_dict[site_name].update({'azimuth': site['Theta']})
                # site_dict[site_name].update({'SolveStatic': site['SolveStatic']})
            site_dict[site_name]['data'][data_type][int(site['Freq#'] - 1)] = site[em_format]
            site_dict[site_name]['errors'][data_type][int(site['Freq#'] - 1)] = site['StdError']
        for site in site_names:
            site_dict[site].update({'azimuth': azimuth,
                                    'periods': periods})
            for component in site_dict[site]['data'].keys():
                vals = site_dict[site]['data'][component]
                site_dict[site]['data'].update({component: np.array(vals)})
                vals = site_dict[site]['errors'][component]
                site_dict[site]['errors'].update({component: np.array(vals)})
        other_info = {'site_names': site_names,
                      'inversion_type': None,
                      'origin': (o_x, o_y),
                      'UTM_zone': UTM_zone}
        return site_dict, other_info

    if file_format.lower() == 'wsinv3dmt':
        return read_ws_data(datafile, site_names, invType)
    elif file_format.lower() == 'modem':
        return read_modem_data(datafile, site_names, invType)
    elif file_format.lower() == 'mare2dem':
        return read_mare2dem_data(datafile, site_names, invType)
    else:
        print('Output format {} not recognized'.format(file_format))
        raise WSFileError(ID='fmt', offender=file_format, expected=('mare2dem',
                                                                    'wsinv3dmt',
                                                                    'ModEM'))


def write_data(data, outfile=None, to_write=None, file_format='WSINV3DMT'):
    #  Writes out the contents of a Data object into format specified by 'file_format'
    #  Currently implemented options include WSINV3DMT and ModEM3D.
    #  Plans to implement OCCAM2D, MARE2DEM, and ModEM2D.
    def write_ws(data, outfile, to_write):
        if '.data' not in outfile:
            outfile = ''.join([outfile, '.data'])
        if to_write == 'all' or to_write is None:
            to_write = ['DATA', 'ERROR', 'ERMAP']
        elif to_write == 'errors':
            to_write = ['ERROR']
        elif to_write == 'errmap':
            to_write = ['ERMAP']
        elif to_write == 'data':
            to_write = ['DATA']
        comps_to_write = data.used_components
        NP = data.NP
        NR = data.NR
        NS = data.NS
        azi = int(data.azimuth)
        ordered_comps = {key: ii for ii, key in enumerate(data.ACCEPTED_COMPONENTS)}
        comps_to_write = sorted(comps_to_write, key=lambda d: ordered_comps[d])
        # theresgottabeabetterway = ('DATA', 'ERROR', 'ERMAP')
        thismeansthat = {'DATA': 'data',
                         'ERROR': 'errors',
                         'ERMAP': 'errmap'}
        if outfile is None:
            print('You have to specify a file!')
            return
        with open(outfile, 'w') as f:
            f.write('{}  {}  {}  {}\n'.format(NS, NP, NR, azi))
            f.write('Station_Location: N-S\n')
            for X in data.locations[:, 0]:
                f.write('{}\n'.format(X))
            f.write('Station_Locations: E-W\n')
            for Y in data.locations[:, 1]:
                f.write('{}\n'.format(Y))
            for this in to_write:
                that = thismeansthat[this]
                for idx, period in enumerate(utils.to_list(data.periods)):
                    f.write(''.join([this, '_Period: ', '%0.5E\n' % float(period)]))
                    for site_name in data.site_names:
                        site = data.sites[site_name]
                        for comp in comps_to_write:
                            try:
                                to_print = getattr(site, that)[comp][idx]
                            except KeyError as er:
                                print('Inside error')
                                print(site.components)
                                print(site.data)
                                raise er
                            if that != 'data':
                                to_print = abs(to_print)
                            # print(to_print[comp][idx])
                            if that == 'errmap':
                                to_print = int(to_print)
                                if to_print == site.OUTLIER_FLAG:
                                    to_print = data.OUTLIER_MAP
                                elif to_print == site.NO_PERIOD_FLAG:
                                    to_print = data.NO_PERIOD_MAP
                                elif to_print == site.NO_COMP_FLAG:
                                    to_print = data.NO_COMP_MAP
                                f.write('{:<5}'.format(min(9999, int(to_print))))
                            else:
                                f.write('{:>14.7E}  '.format(to_print))
                            # for point in getattr(site, that)[comp]:
                        f.write('\n')

    def write_ModEM3D(data, out_file):
        units = []
        data_type = []
        temp_inv_type = []
        with open(out_file, 'w') as f:
            title = '# Written using pyMT. UTM Zone: {}\n' + \
                    '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) ' + \
                    'Component Real Imag Error\n'.format(data.UTM_zone)
            if data.inv_type == 1:
                data_type.append('> Full_Impedance\n')
                units.append('> Ohm\n')
                temp_inv_type.append(1)
            elif data.inv_type == 2:
                data_type.append('> Off_Diagonal_Impedance\n')
                units.append('> Ohm\n')
                temp_inv_type.append(2)
            elif data.inv_type == 3:
                data_type.append('> Full_Vertical_Components\n')
                units.append('> []\n')
                temp_inv_type.append(3)
            elif data.inv_type == 4:
                data_type.append('> Off_Diagonal_Impedance\n')
                units.append('> Ohm\n')
                data_type.append('> Full_Vertical_Components\n')
                units.append('> []\n')
                temp_inv_type.append(2)
                temp_inv_type.append(3)
            elif data.inv_type == 5:
                data_type.append('> Full_Impedance\n')
                units.append('> Ohm\n')
                data_type.append('> Full_Vertical_Components\n')
                units.append('> []\n')
                temp_inv_type.append(1)
                temp_inv_type.append(3)
            print(data.inv_type)
            print(temp_inv_type)
            for data_type_string, inv_type, unit in zip(data_type, temp_inv_type, units):
                f.write(title)
                f.write(data_type_string)
                f.write('> exp(-i\\omega t)\n')
                f.write(unit)
                f.write('> {}\n'.format(data.azimuth))
                f.write('> 0.0 0.0 0.0\n')
                f.write('> {} {}\n'.format(data.NP, data.NS))
                data.inv_type = inv_type
                components_to_write = [component for component in data.used_components
                                       if 'i' not in component.lower()]
                for site_name in data.site_names:
                    site = data.sites[site_name]
                    for jj, period in enumerate(data.periods):
                        for component in components_to_write:
                            component_code = component[:3]
                            if 'T' in component.upper():
                                component_code = component_code[0] + component_code[2]
                            Z_real = site.data[component][jj]
                            Z_imag = site.data[component[:3] + 'I'][jj]
                            X, Y, Z = site.locations['X'], site.locations['Y'], site.locations.get('elev', 0)
                            Lat, Long = site.locations.get('Lat', 0), site.locations.get('Long', 0)
                            f.write(' '.join(['{:>14.7E} {}',
                                              '{:>8.3f} {:>8.3f}',
                                              '{:>15.3f} {:>15.3f} {:>15.3f}',
                                              '{:>6} {:>14.7E} {:>14.7E}',
                                              '{:>14.7E}\n']).format(
                                    period, site_name,
                                    Lat, Long,
                                    X, Y, Z,
                                    component_code.upper(), Z_real, Z_imag,
                                    site.used_error[component][jj]))

    def write_MARE2DEM(data, out_file):
        data_type_lookup = {'RhoZXX': 101,
                            'PhsZXX': 102,
                            'RhoZXY': 103,
                            'PhsZXY': 104,
                            'RhoZYX': 105,
                            'PhsZYX': 106,
                            'RhoZYY': 107,
                            'PhsZYY': 108,
                            'ZXXR': 111,
                            'ZXXI': 112,
                            'ZXYR': 113,
                            'ZXYI': 114,
                            'ZYXR': 115,
                            'ZYXI': 116,
                            'ZYYR': 117,
                            'ZYYI': 118,
                            'log10RhoZXX': 121,
                            'log10RhoZXY': 123,
                            'log10RhoZYX': 125,
                            'log10RhoZYY': 127,
                            'TZYR': 133,
                            'TZYI': 134}
        strike = (data.azimuth - 90) % 180
        frequencies = [(1 / p) for p in data.periods]
        with open(outfile, 'w') as f:
            f.write('Format:\tEMData_2.2\n')
            f.write('UTM of x, y origin (UTM zone, N, E, 2D strike): {:>10}{:>10}{:>10}{:>10}\n'.format(
                    data.UTM_zone, data.origin[0], data.origin[1], strike))
            f.write('# MT Frequencies: {}\n'.format(data.NP))
            for freq in frequencies:
                f.write('  {}\n'.format(freq))
            f.write('# MT Receivers: {}\n'.format(data.NS))
            f.write('!{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
                    'X', 'Y', 'Z', 'Theta', 'Alpha', 'Beta', 'Length', 'SolveStatic', 'Name'))
            for site in data.site_names:
                f.write(' {:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
                    data.sites[site].locations['X'],
                    data.sites[site].locations['Y'],
                    data.sites[site].locations['elev'],
                    data.sites[site].azimuth,
                    getattr(data.sites[site], 'alpha', 0),
                    getattr(data.sites[site], 'beta', 0),
                    getattr(data.sites[site], 'dipole_length', 0),
                    getattr(data.sites[site], 'solve_static', 0),
                    site))
            f.write('# Data: {}\n'.format(data.NS * data.NP * len(data.used_components)))
            f.write('!{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
                    'Type', 'Freq#', 'Tx#', 'Rx#', 'Data', 'StdError'))
            for ii, site in enumerate(data.site_names):
                for jj, frequency in enumerate(frequencies):
                    for component in data.used_components:
                        f.write('{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
                                data_type_lookup[component],
                                jj + 1,
                                1,
                                ii + 1,
                                data.sites[site].data[component][jj],
                                data.sites[site].used_error[component][jj]))

    if file_format.lower() == 'wsinv3dmt':
        write_ws(data, outfile, to_write)
    elif file_format.lower() == 'modem':
        write_ModEM3D(data, outfile)
    elif file_format.lower() == 'mare2dem':
        write_MARE2DEM(data, outfile)
    else:
        print('Output file format {} not recognized'.format(file_format))


def write_response(data, outfile=None):
    NP = data.NP
    NR = data.NR
    NS = data.NS
    azi = int(data.azimuth)
    ordered_comps = {key: ii for ii, key in enumerate(data.ACCEPTED_COMPONENTS)}
    comps_to_write = data.used_components
    comps_to_write = sorted(comps_to_write, key=lambda d: ordered_comps[d])
    if outfile is None:
        print('You have to specify a file!')
        return
    with open(outfile, 'w') as f:
        f.write('{}  {}  {}  {}\n'.format(NS, NP, NR, azi))
        f.write('Station_Location: N-S\n')
        for X in data.locations[:, 0]:
            f.write('{}\n'.format(X))
        f.write('Station_Locations: E-W\n')
        for Y in data.locations[:, 1]:
            f.write('{}\n'.format(Y))
        for idx, period in enumerate(utils.to_list(data.periods)):
            f.write(''.join(['DATA_Period: ', '%0.5E\n' % float(period)]))
            for site_name in data.site_names:
                site = data.sites[site_name]
                for comp in comps_to_write:
                    to_print = getattr(site, 'data')[comp][idx]
                    f.write('{:>14.7E}  '.format(to_print))
                    # for point in getattr(site, that)[comp]:
                f.write('\n')


def write_errors(data, outfile=None):
    NP = data.NP
    NR = data.NR
    NS = data.NS
    azi = int(data.azimuth)
    ordered_comps = {key: ii for ii, key in enumerate(data.ACCEPTED_COMPONENTS)}
    comps_to_write = data.used_components
    comps_to_write = sorted(comps_to_write, key=lambda d: ordered_comps[d])
    if outfile is None:
        print('You have to specify a file!')
        return
    with open(outfile, 'w') as f:
        f.write('{}  {}  {}  {}\n'.format(NS, NP, NR, azi))
        f.write('Station_Location: N-S\n')
        for X in data.locations[:, 0]:
            f.write('{}\n'.format(X))
        f.write('Station_Locations: E-W\n')
        for Y in data.locations[:, 1]:
            f.write('{}\n'.format(Y))
        for idx, period in enumerate(utils.to_list(data.periods)):
            f.write(''.join(['ERROR_Period: ', '%0.5E\n' % float(period)]))
            for site_name in data.site_names:
                site = data.sites[site_name]
                for comp in comps_to_write:
                    to_print = getattr(site, 'used_error')[comp][idx]
                    f.write('{:>14.7E}  '.format(to_print))
                    # for point in getattr(site, that)[comp]:
                f.write('\n')


def model_to_vtk(model, outfile=None, origin=None, UTM=None, azi=0, sea_level=0):
    """
    Write model to VTK ascii format
    VTK format has X running west-east, Y running south-north, so model dimensions
    need to be swapped.
    Model origin should be archived throughout the process, i.e. stuffed into model
    or data files so it can be recovered at any time. Positive Z goes UP, so model needs to
    be flipped in Z.
    This also can accept a data file so that a seperate VTK file is written with site
    locations. Because of the fact that 2 files can be read, you should also look for
    UTM coordinates in both, if necessary / possible.
    Want this process to be as easy as possible, so there should probably be
    several possible ways of reading. You could specify directly the model and data files,
    specify a file containing the names of the files, you can specify nothing and it could
    look for the model with the best RMS and use that, or there could just be a browser.
    For now I'll just have to do things manually until I get it all built and moved to the
    cluster.
    Usage on the cluster will probably have to be a call to a script which creates instances
    of the data and model from the passed files, then passes these to this function.
    That might be overkill though, so another option might just to be to have a lightweight
    version of this that just does what is necessary to spit out the VTK file.
    All of Model, Data, and Dataset should probably have wrapper methods for this though that
    just redirect the data to here.
    """
    def prep_model(model):
        values = copy.deepcopy(model)
        tmp = values.vals
        values.vals = np.swapaxes(tmp, 0, 1)
        NX, NY, NZ = values.vals.shape
        values._dx, values._dy = values._dy, values._dx
        values._dx = [x + ox for x in values._dx]
        values._dy = [y + oy for y in values._dy]
        values._dz = [-z + sea_level for z in values._dz]
        return values
    # print(model.resolution)
    errmsg = ''
    ox, oy = (0, 0)
    if origin:
        try:
            ox, oy = origin
        except TypeError:
            errmsg = '\n'.join(errmsg, 'Model origin must be properly specified.')
    else:
        ox, oy = model.origin
    if not UTM:
        try:
            UTM = model.UTM_zone
        except AttributeError:
            errmsg.append(['ERROR: UTM must be specified either in function call or in model'])
    if errmsg:
        print('\n'.join(errmsg))
        resp = input('Do you want to continue? (y/n)')
        if resp.lower() == 'n':
            return
    version = '# vtk DataFile Version 3.0\n'
    modname = os.path.basename(model.file)
    if not outfile:
        outfile = ''.join([modname, '_model.vtk'])
    else:
        outfile = '_'.join([outfile, 'model.vtk'])
    values = prep_model(model)
    if model.resolution:
        # This creates separate copies of the model object and sticks the resolution values into the vals
        # attributes
        # Is a very roundabout method. All resolution information
        # should be availabe in the original model object
        print('Model.resolution exists')
        raw_resolution = copy.deepcopy(model)
        raw_resolution.vals = model.resolution
        if np.max(raw_resolution.vals) > 1:  # Invert values if not already done
            raw_resolution.vals = 1 / raw_resolution.vals
        raw_resolution.vals[raw_resolution.vals > 1e10] = 1e10
        modified_resolution = copy.deepcopy(raw_resolution)
        modified_resolution = utils.normalize(modified_resolution)
        modified_resolution.vals[modified_resolution.vals > 1] = 1
        raw_resolution = prep_model(raw_resolution)
        modified_resolution = prep_model(modified_resolution)
        scalars = ('Resistivity', 'Raw_Resolution', 'Modified_Resolution')
        to_write = (values, raw_resolution, modified_resolution)
    else:
        scalars = ['Resistivity']
        to_write = [values]
    # if azi:
    #     use_rot = True
    #     X, Y = np.meshgrid(values.dx, values.dy)
    #     locs = np.transpose(np.array((np.ndarray.flatten(X), np.ndarray.flatten(Y))))
    #     locs = utils.rotate_locs(locs, azi=-azi)
    # else:
    # use_rot = False
    # values.vals = np.reshape(values.vals, [NX * NY * NZ], order='F')
    NX, NY, NZ = values.vals.shape
    with open(outfile, 'w') as f:
        f.write(version)
        f.write('{}   UTM: {} \n'.format(modname, UTM))
        f.write('ASCII\n')
        f.write('DATASET RECTILINEAR_GRID\n')
        f.write('DIMENSIONS {} {} {}\n'.format(NX + 1, NY + 1, NZ + 1))
        for dim in ('x', 'y', 'z'):
            f.write('{}_COORDINATES {} float\n'.format(dim.upper(),
                                                       len(getattr(values, ''.join(['_d', dim])))))
            gridlines = getattr(values, ''.join(['_d', dim]))
            for edge in gridlines:
                f.write('{} '.format(str(edge)))
            f.write('\n')
            # for ii in range(getattr(values, ''.join(['n', dim]))):
            #     midpoint = (gridlines[ii] + gridlines[ii + 1]) / 2
            #     f.write('{} '.format(str(midpoint)))
            # f.write('\n')
        f.write('POINT_DATA {}\n'.format((NX + 1) * (NY + 1) * (NZ + 1)))

        for ii, value in enumerate(scalars):

            f.write('SCALARS {} float\n'.format(value))
            f.write('LOOKUP_TABLE default\n')
            # print(len())
            for iz in range(NZ + 1):
                for iy in range(NY + 1):
                    for ix in range(NX + 1):
                            xx = min([ix, NX - 1])
                            yy = min([iy, NY - 1])
                            zz = min([iz, NZ - 1])
                            f.write('{}\n'.format(to_write[ii].vals[xx, yy, zz]))


def sites_to_vtk(data, origin=None, outfile=None, UTM=None, sea_level=0):
    errmsg = ''
    ox, oy = (0, 0)
    if isinstance(origin, str):
        pass
    else:
        try:
            ox, oy = origin
        except TypeError:
            errmsg = '\n'.join([errmsg, 'Model origin must be properly specified.'])
    if not UTM:
        errmsg = '\n'.join(['ERROR: UTM must be specified either in function call or in model'])
    if errmsg:
        print('\n'.join(errmsg))
        return
    version = '# vtk DataFile Version 3.0\n'
    if not outfile:
        print('You must specify the output file name')
        return
    if '.vtk' not in outfile:
        outfile = ''.join([outfile, '_sites.vtk'])
    xlocs = data.locations[:, 1] + origin[0]
    ylocs = data.locations[:, 0] + origin[1]
    ns = len(xlocs)
    with open(outfile, 'w') as f:
        f.write(version)
        f.write('UTM: {} \n'.format(UTM))
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        # f.write('DIMENSIONS {} {} {} \n'.format(ns, ns, 1))
        f.write('POINTS {} float\n'.format(ns))
        for ix, iy in zip(xlocs, ylocs):
            f.write('{} {} {}\n'.format(ix, iy, sea_level))
        f.write('POINT_DATA {}\n'.format(ns))
        f.write('SCALARS dummy float\n')
        f.write('LOOKUP_TABLE default\n')
        for ii in range(ns):
            f.write('{}\n'.format(999999))


def read_freqset(path):
    freqset = PATH_CONNECTOR.join([path, 'freqset'])
    with open(freqset, 'r') as f:
        lines = f.readlines()
        # nf = int(lines[0])
        periods = [float(x) for x in lines[1:]]
        # if len(periods) != nf:
        #     print('Quoted number of periods, {}, is not equal to the actual number, {}'.format(
        #           nf, len(periods)))
        #     while True:
        #         resp = input('Continue anyways? (y/n)')
        #         if resp not in ('yn'):
        #             print('Try again.')
        #         else:
        #             break
    return periods


def write_list(data, outfile):
    if '.lst' not in outfile:
        outfile = ''.join([outfile, '.lst'])
    with open(outfile, 'w') as f:
        f.write('{}\n'.format(len(data.site_names)))
        for site in data.site_names[:-1]:
            f.write('{}\n'.format(''.join([site, '.dat'])))
        f.write('{}'.format(''.join([data.site_names[-1], '.dat'])))


def write_model(model, outfile):
    if '.model' not in outfile:
        outfile = ''.join([outfile, '.model'])
    is_half_space = int(model.is_half_space())
    with open(outfile, 'w') as f:
        f.write('{}\n'.format(outfile))
        f.write('{} {} {} {}\n'.format(model.nx, model.ny, model.nz, is_half_space))
        for x in model.xCS:
            f.write('{:<10.7f}  '.format(x))
        f.write('\n')
        for y in model.yCS:
            f.write('{:<10.7f}  '.format(y))
        f.write('\n')
        for z in model.zCS:
            f.write('{:<10.7f}  '.format(z))
        f.write('\n')
        if is_half_space:
            f.write(str(model.vals[0, 0, 0]))
        else:
            for zz in range(model.nz):
                for yy in range(model.ny):
                    for xx in range(model.nx):
                        # f.write('{:<10.7E}\n'.format(model.vals[model.nx - xx - 1, yy, zz]))
                        f.write('{:<10.7f}\n'.format(model.vals[model.nx - xx - 1, yy, zz]))


def read_occam_data(datafile):
    with open(datafile, 'r') as f:
        lines = f.readlines()
        for ii, line in enumerate(lines):
            if 'sites' in line.lower():
                NS = int(line.split(':')[1].strip())
                site_line = ii + 1
            if 'offsets' in line.lower():
                off_line = ii + 1
            if 'frequencies' in line.lower():
                NF = int(line.split(':')[1].strip())
                freq_line = ii + 1
                break
        frequencies = []
        offsets = []
        for line in lines[off_line: off_line + NS]:
            if 'frequencies' in line.lower():
                break
            else:
                offsets.append(line)
        for ii, line in enumerate(lines[freq_line: freq_line + NF]):
            if 'DATA' in line.upper():
                ND = int(line.split(':')[1].strip())
                data_line = ii + 2
                break
            else:
                frequencies.append(line)
        frequencies = utils.flatten_list([x.split() for x in frequencies])
        # frequencies = [item for sublist in frequencies for item in sublist]
        # frequencies = [float(x) for x in frequencies]
        frequencies = np.array([float(f) for f in frequencies])
        sites = lines[site_line: site_line + NS]
        sites = [site.strip() for site in sites]
        offsets = utils.flatten_list([x.split() for x in offsets])
        offsets = [float(x) for x in offsets]
        all_data = np.zeros((ND, 5))
        for ii, line in enumerate(lines[data_line:]):
            all_data[ii, :] = [float(x) for x in line.split()]
        sites_dict = {}
        for ii, site in enumerate(sites):
            site_data = {'ZXYR': [], 'ZXYI': [], 'ZYXR': [], 'ZYXI': []}
            rhoxy = all_data[np.bitwise_and(all_data[:, 2] == 1, all_data[:, 0] == ii + 1), 3]
            rhoyx = all_data[np.bitwise_and(all_data[:, 2] == 5, all_data[:, 0] == ii + 1), 3]
            phaxy = all_data[np.bitwise_and(all_data[:, 2] == 2, all_data[:, 0] == ii + 1), 3]
            phayx = all_data[np.bitwise_and(all_data[:, 2] == 6, all_data[:, 0] == ii + 1), 3]
            site_data['ZXYR'], site_data['ZXYI'] = utils.convert2impedance(rhoxy,
                                                                           phaxy,
                                                                           1 / frequencies,
                                                                           'xy')
            site_data['ZYXR'], site_data['ZYXI'] = utils.convert2impedance(rhoyx,
                                                                           phayx,
                                                                           1 / frequencies,
                                                                           'yx')
            site_errs = {'ZXYR': np.zeros((NF, 1)),
                         'ZXYI': np.zeros((NF, 1)),
                         'ZYXR': np.zeros((NF, 1)),
                         'ZYXI': np.zeros((NF, 1))}
            site_errmap = {'ZXYR': np.ones((NF, 1)),
                           'ZXYI': np.ones((NF, 1)),
                           'ZYXR': np.ones((NF, 1)),
                           'ZYXI': np.ones((NF, 1))}
            locations = {'X': 0, 'Y': offsets[ii]}
            sites_dict.update({site: {
                               'data': site_data,
                               'errors': site_errs,
                               'errmap': site_errmap,
                               'periods': 1 / frequencies,
                               'locations': locations,
                               'azimuth': 0,
                               'errFloorZ': None,
                               'errFloorT': None}
                               })
    return sites_dict, sites


def read_occam_response(respfile, data, sites):
    all_data = np.loadtxt(respfile)
    response = copy.deepcopy(data)
    for ii, site in enumerate(sites):
        rhoxy = all_data[np.bitwise_and(all_data[:, 2] == 1, all_data[:, 0] == ii + 1), 5]
        rhoyx = all_data[np.bitwise_and(all_data[:, 2] == 5, all_data[:, 0] == ii + 1), 5]
        phaxy = all_data[np.bitwise_and(all_data[:, 2] == 2, all_data[:, 0] == ii + 1), 5]
        phayx = all_data[np.bitwise_and(all_data[:, 2] == 6, all_data[:, 0] == ii + 1), 5]
        response[site]['data']['ZXYR'],
        response[site]['data']['ZXYI'] = utils.convert2impedance(rhoxy,
                                                                 phaxy,
                                                                 data[site]['periods'],
                                                                 'xy')
        response[site]['data']['ZYXR'],
        response[site]['data']['ZYXI'] = utils.convert2impedance(rhoyx,
                                                                 phayx,
                                                                 data[site]['periods'],
                                                                 'yx')
        return response


# def occam2ws(respfile, datafile, listfile=None, datpath=None):

