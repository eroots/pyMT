from pyMT.WSExceptions import WSFileError
import pyMT.utils as utils
import numpy as np
import os
import copy
import re


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
                if ret.lower() not in expected:
                    print('That is not an option. Try again.')
                else:
                    return ret.lower()
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
    if not NR:
        NR = 0
    if (NR == 8 and not invType) or (invType == 1):
        comps = [possible[0:8], tuple(range(8))]
    elif (NR == 12 and not invType) or (invType == 5):
        comps = [possible, tuple(range(12))]
    elif (NR == 4 and not invType) and (invType == 2):
        comps = [possible[2:5], tuple(range(4))]
    elif invType == 3:
        comps = [possible[8:], tuple(range(4))]
    elif invType == 4:
        comps = [possible[2, 3, 4, 5, 8, 9, 10, 11],
                 tuple(range(8))]
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
        NX, NY, NZ, MODTYPE = [int(h) for h in header.split()]
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
        return mod


def read_raw_data(site_names, datpath=''):
    RAW_COMPONENTS = ('ZXX', 'ZXY',
                      'ZYX', 'ZYY',
                      'TZX', 'TZY')
    """Summary

    Args:
        site_names (TYPE): Description
        datpath (str, optional): Description

    Returns:
        TYPE: Description
    """
    siteData = {}
    if os.name is 'nt':
        connector = '\\'
    else:
        connector = '/'
    if datpath:
        path = datpath + connector
    else:
        path = './'
    long_origin = 999
    for site in site_names:
        file = ''.join([path, site, '.dat'])
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
                for ii, comp in comps:
                    ns = int(lines[ii + 1])
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

                siteData.update({site:
                                 {'data': siteData_dict,
                                  'errors': siteError_dict,
                                  'locations': siteLoc_dict,
                                  'periods': periods,
                                  'azimuth': siteLoc_dict['azi']
                                  }
                                 })
        except FileNotFoundError as e:
            raise(WSFileError(ID='fnf', offender=file))
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


def read_data(datafile='', site_names='', filetype='data', invType=None):
    """Summary

    Args:
        datafile (str, optional): Description
        site_names (str, optional): Description
        filetype (str, optional): Description
        invType (None, optional): Description

    Returns:
        TYPE: Description
    """
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
                site_names = [str(x) for x in list(range(1, NS + 1))]
            if NS != len(site_names):
                raise(WSFileError(ID='int', offender=datafile,
                                  extra='Number of sites in data file not equal to list file'))
            # Components is a pair of (compType, Nth item to grab)
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
                              'azimuth': azi
                              }
                              })
    except FileNotFoundError as e:
        raise(WSFileError(ID='fnf', offender=datafile)) from None
    return sites


def write_data(data, outfile=None):
    if '.data' not in outfile:
        outfile = ''.join([outfile, '.data'])
    comps_to_write = data.used_components
    NP = data.NP
    NR = data.NR
    NS = data.NS
    azi = int(data.azimuth)
    ordered_comps = {key: ii for ii, key in enumerate(data.ACCEPTED_COMPONENTS)}
    comps_to_write = sorted(comps_to_write, key=lambda d: ordered_comps[d])
    theresgottabeabetterway = ('DATA', 'ERROR', 'ERMAP')
    thismeansthat = {'DATA': 'data',
                     'ERROR': 'errors',
                     'ERMAP': 'errmap'}
    if outfile is None:
        print('You have to specify a file!')
        return
    with open(outfile, 'w') as f:
        f.write('{}  {}  {}  {}\n'.format(NS, NP, NR, azi))
        f.write('Station_Location: N-S\n')
        print(type(data))
        print(type(data.locations))
        for X in data.locations[:, 0]:
            f.write('{}\n'.format(X))
        f.write('Station_Locations: E-W\n')
        for Y in data.locations[:, 1]:
            f.write('{}\n'.format(Y))
        for this in theresgottabeabetterway:
            that = thismeansthat[this]
            for idx, period in enumerate(utils.to_list(data.periods)):
                f.write(''.join([this, '_Period: ', '%0.5E\n' % float(period)]))
                for site_name in data.site_names:
                    site = data.sites[site_name]
                    for comp in comps_to_write:
                        to_print = getattr(site, that)[comp][idx]
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
    errmsg = ''
    ox, oy = (0, 0)
    if origin:
        try:
            ox, oy = origin
        except:
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
    values = copy.deepcopy(model)
    tmp = values.vals
    values.vals = np.swapaxes(tmp, 0, 1)
    NX, NY, NZ = values.vals.shape
    values.dx, values.dy = values.dy, values.dx
    values.dx = [x + ox for x in values.dx]
    values.dy = [y + oy for y in values.dy]
    values.dz = [-z + sea_level for z in values.dz]
    # if azi:
    #     use_rot = True
    #     X, Y = np.meshgrid(values.dx, values.dy)
    #     locs = np.transpose(np.array((np.ndarray.flatten(X), np.ndarray.flatten(Y))))
    #     locs = utils.rotate_locs(locs, azi=-azi)
    # else:
    # use_rot = False
    # values.vals = np.reshape(values.vals, [NX * NY * NZ], order='F')
    with open(outfile, 'w') as f:
        f.write(version)
        f.write('{}   UTM: {} \n'.format(modname, UTM))
        f.write('ASCII\n')
        f.write('DATASET RECTILINEAR_GRID\n')
        f.write('DIMENSIONS {} {} {}\n'.format(NX + 1, NY + 1, NZ + 1))
        for dim in ('x', 'y', 'z'):
            f.write('{}_COORDINATES {} float\n'.format(dim.upper(), 1 +
                                                       getattr(values, ''.join(['n', dim]))))
            gridlines = getattr(values, ''.join(['d', dim]))
            for edge in gridlines:
                f.write('{} '.format(str(edge)))
            f.write('\n')
            # for ii in range(getattr(values, ''.join(['n', dim]))):
            #     midpoint = (gridlines[ii] + gridlines[ii + 1]) / 2
            #     f.write('{} '.format(str(midpoint)))
            # f.write('\n')
        f.write('POINT_DATA {}\n'.format((NX + 1) * (NY + 1) * (NZ + 1)))
        f.write('SCALARS Resistivity float\n')
        f.write('LOOKUP_TABLE default\n')
        # print(len())
        for iz in range(NZ + 1):
            for iy in range(NY + 1):
                for ix in range(NX + 1):
                        xx = min([ix, NX - 1])
                        yy = min([iy, NY - 1])
                        zz = min([iz, NZ - 1])
                        f.write('{}\n'.format(values.vals[xx, yy, zz]))


def sites_to_vtk(data, origin=None, outfile=None, UTM=None, sea_level=0):
    errmsg = ''
    ox, oy = (0, 0)
    if isinstance(origin, str):
        pass
    else:
        try:
            ox, oy = origin
        except:
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


def read_freqset(freqset='freqset'):
    with open(freqset, 'r') as f:
        lines = f.readlines()
        nf = int(lines[0])
        periods = [float(x) for x in lines]
        if len(periods) != nf:
            print('Quoted number of periods, {}, is not equal to the actual number, {}'.format(
                  nf, len(periods)))
            while True:
                resp = input('Continue anyways? (y/n)')
                if resp not in ('yn'):
                    print('Try again.')
                else:
                    break


def write_list(data, outfile):
    if '.lst' not in outfile:
        outfile = ''.join([outfile, '.lst'])
    with open(outfile, 'w') as f:
        f.write('{}\n'.format(len(data.site_names)))
        for site in data.site_names[:-1]:
            f.write('{}\n'.format(''.join([site, '.dat'])))
        f.write('{}'.format(''.join([data.site_names[-1], '.dat'])))
