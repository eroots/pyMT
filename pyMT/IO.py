from pyMT.WSExceptions import WSFileError
import pyMT.utils as utils
import numpy as np
import os
import copy
import re
import shapefile
import datetime
from collections import OrderedDict

REMOVE_FLAG = 1234567


INVERSION_TYPES = {1: ('ZXXR', 'ZXXI',  # 1-5 are WS formats
                       'ZXYR', 'ZXYI',
                       'ZYXR', 'ZYXI',
                       'ZYYR', 'ZYYI'),
                   2: ('ZXYR', 'ZXYI',
                       'ZYXR', 'ZYXI'),
                   3: ('TZXR', 'TZXI',
                       'TZYR', 'TZYI'),
                   4: ('ZXYR', 'ZXYI',
                       'ZYXR', 'ZYXI',
                       'TZXR', 'TZXI',
                       'TZYR', 'TZYI'),
                   5: ('ZXYR', 'ZXYI',
                       'ZYXR', 'ZYXI',
                       'TZXR', 'TZXI',
                       'TZYR', 'TZYI',
                       'ZXXR', 'ZXXI',
                       'ZYYR', 'ZYYI'),
                   6: ('PTXX', 'PTXY',  # 6 is ModEM Phase Tensor inversion
                       'PTYX', 'PTYY'),
                   7: ('PTXX', 'PTXY',
                       'PTYX', 'PTYY',
                       'TZXR', 'TZXI',
                       'TZYR', 'TZYI'),
                   8: ('ZXYR', 'ZXYI'),  # 2-D ModEM inversion of TE mode
                   9: ('ZYXR', 'ZYXI'),  # 2-D ModEM inversion of TM mode
                   10: ('ZXYR', 'ZXYI',  # 2-D ModEM inversion of TE+TM modes
                        'ZYXR', 'ZYXI'),
                   11: ('RhoZXX', 'PhszXX',  # 7-15 are reserved for MARE2DEM inversions
                        'RhoZXY', 'PhszXY',
                        'RhoZYX', 'PhszYX',
                        'RhoZYY', 'PhszYY'),
                   12: ('RhoZXY', 'PhsZXY',
                        'RhoZYX', 'PhsZYX'),
                   13: ('TZYR', 'TZYI'),
                   14: ('RhoZXY', 'PhsZXY',
                        'RhoZYX', 'PhsZYX',
                        'TZYR', 'TZYI'),
                   15: ('RhoZXX', 'PhsZXX',
                        'RhoZXY', 'PhsZXY',
                        'RhoZYX', 'PhsZYX',
                        'RhoZYY', 'PhsZYY',
                        'TZYR', 'TZYI'),
                   16: ('log10RhoZXX', 'PhsZXX',
                        'log10RhoXY', 'PhsXY',
                        'log10RhoYX', 'PhsYX',
                        'log10RhoYY', 'PhsYY'),
                   17: ('log10RhoZXY', 'PhsZXY',
                        'log10RhoZYX', 'PhsZYX'),
                   18: ('log10RhoZXY', 'PhsZXY',
                        'log10RhoZYX', 'PhsZYX',
                        'TZYR', 'TZYI'),
                   19: ('log10RhoZXX', 'PhsZXX',
                        'log10RhoZXY', 'PhsZXY',
                        'log10RhoZYX', 'PhsZYX',
                        'log10RhoZYY', 'PhsZYY',
                        'TZYR', 'TZYI')}


if os.name == 'nt':
    PATH_CONNECTOR = '\\'
else:
    PATH_CONNECTOR = '/'


def debug_print(items, file):
    with open(file, 'a+') as f:
        datetime_obj = datetime.datetime.now()
        f.write(str(datetime_obj) + '\n')
        f.write('{}\n'.format(items))


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


def read_covariance(cov_file):
    # Currently does not support actual smoothing exceptions, just air and ocean masks.
    with open(cov_file, 'r') as f:
        for ii in range(16):
            f.readline()
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line]
        nx, ny, nz = [int(x) for x in lines[0].split()]
        sigma_x = [float(x) for x in lines[1].split()]
        sigma_y = [float(x) for x in lines[2].split()]
        sigma_z = [float(x) for x in lines[3].split()]
        num_smooth = int(lines[4])
        num_exception = int(lines[5])
        # A standard covariance file would end here.
        try:
            covlines = lines[6:]
            covlines = [l.strip() for l in covlines]
            covlines = [l for l in covlines if l]
            cov_exceptions = np.ones((nx, ny, nz))
            for ii in range(nz):
                idx1, idx2 = [int(x) for x in covlines[ii + ii * nx].strip().split()]
                for jj in range(nx):
                    cov_exceptions[model.nx - jj - 1, :, idx1-1:idx2] = np.expand_dims(np.array([int(x) for x in covlines[jj + 1 + ii * (nx + 1)].split()]), axis=1)
        except IndexError:
            cov_expections = 0
        return nx, ny, nz, sigma_x, sigma_y, sigma_z, num_smooth, cov_exceptions


def read_freqset(path='./'):
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


def read_model(modelfile='', file_format='modem3d'):

    def read_em3dani(modelfile):
        # Not fully written - probably won't work for general anisotropy yet.
        with open(modelfile, 'r') as f:
            mod = {'xCS': [], 'yCS': [], 'zCS': [], 'vals': [], 
                  'rho_x': [], 'rho_y': [], 'rho_z': [], 'zAir': [],
                  'strike': [], 'dip': [], 'slant': []}
            counter = 0
            header = '#'
            while header.strip().startswith('#'):
                header = next(f)
            NX = int(header.split(':')[1])
            # print(next(f))
            # xCS = []
            for direction in ('xCS', 'yCS', 'zAir', 'zCS'):
                vals = []
                line = f.readline()
                while True:
                    try:
                        vals.append([float(x) for x in line.split()])
                        line = f.readline()
                    except ValueError:
                        break
                mod[direction] = [item for sublist in vals for item in sublist]
            val_type = line.split(':')[1].strip()
            model_type = next(f).split(':')[1].strip()
            NY = len(mod['yCS'])
            NZ = len(mod['zCS'])
            line = next(f)
            if 'Anisotropy' in line:
                rhos = ('rho_x', 'rho_y', 'rho_z')
                line = next(f)
            else:
                rhos = ['rho_x']
            for rho in rhos:
                line = next(f)
                vals = []
                counter = 0
                while len(line.split(':')) == 1:
                    vals.append([float(x) for x in line.split()])
                    counter += 1
                    line = next(f)
                vals = np.array([item for sublist in vals for item in sublist])
                if val_type.lower() == 'conductivity':
                    vals = 1 / vals
                if model_type.lower() == 'log':
                    vals = 10 ** vals
                mod[rho] = np.reshape(np.array(vals), [NX, NY, NZ], order='F') # flipud?
            mod['vals'] = copy.deepcopy(mod['rho_x'])

            return mod, False

    def read_3d(modelfile):
        with open(modelfile, 'r') as f:
            mod = {'xCS': [], 'yCS': [], 'zCS': [], 'vals': []}
            # Skip any comment lines at the top
            counter = 0
            while True:
            # header = next(f)
                header = next(f)
                if not header.strip().startswith('#') and counter:
                    break
                counter += 1
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
        return mod, loge_flag

    def read_2d(modelfile):
        with open(modelfile, 'r') as f:
            mod = {'xCS': [100] * 10, 'yCS': [], 'zCS': [], 'vals': []}
            # while True:
            header = next(f)
            # if header[0] != '#':
            # break
            NX = 10
            NY, NZ, MODTYPE = [h for h in header.split()]
            NY, NZ = int(NY), int(NZ)
            loge_flag = False
            if MODTYPE == 'LOGE':
                loge_flag = True
            lines = f.readlines()
            lines = [x.split() for x in lines]
            lines = [item for sublist in lines for item in sublist]
            lines = [float(x) for x in lines]
            mod['yCS'] = lines[:NY]
            # mod['yCS'] = lines[NX: NX + NY]
            mod['zCS'] = lines[NY: NY + NZ]
            # if MODTYPE == 1:
                # mod['vals'] = np.ones([NX, NY, NZ]) * lines[NX + NY + NZ]
            # else:
                # vals = lines[NX + NY + NZ: ]
            mod['vals'] = np.reshape(np.array(lines[NY + NZ: NY + NZ + NY * NZ]),
                                     [NY, NZ], order='F')
            mod['vals'] = np.tile(mod['vals'][np.newaxis, :, :], [NX, 1, 1])
        return mod, loge_flag
   
    def read_mt3dani(modelfile, n_param=3):
        with open(modelfile, 'r') as f:
            mod = {'xCS': [], 'yCS': [], 'zCS': [], 'vals': [], 
                  'rho_x': [], 'rho_y': [], 'rho_z': [],
                  'strike': [], 'dip': [], 'slant': []}
            # Skip any comment lines at the top
            counter = 0
            while True:
            # header = next(f)
                header = next(f)
                if not header.strip().startswith('#') and counter:
                    break
                counter += 1
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
                param = ['rho_x', 'rho_y', 'rho_z', 'strike', 'dip', 'slant']
                for pp in range(n_param):
                    idx1 = NX + NY + NZ + (NX*NY*NZ*pp)
                    idx2 = NX + NY + NZ + (NX*NY*NZ*(pp+1))
                    vals = np.flipud(np.reshape(np.array(lines[idx1:idx2]), [NX, NY, NZ], order='F'))
                    if loge_flag == True:
                        vals = np.exp(vals)
                    mod.update({param[pp]: vals})
                mod['vals'] = copy.deepcopy(mod['rho_x'])
                loge_flag = False
        return mod, loge_flag

    try:
        if not modelfile:
            print('No model to read')
            return None
        if modelfile[-4:] == 'zani':
            print('Reading MT3DANI model')
            file_format = 'mt3dani'
        if file_format.lower() == 'modem3d':
            try:
                mod, loge_flag = read_3d(modelfile)
                dimensionality = '3d'
            except ValueError:
                try:
                    print('Model not in ModEM3D format. Trying 2D')
                    mod, loge_flag = read_2d(modelfile)
                    dimensionality = '2d'
                except ValueError:
                    print('Not in ModEM format. Trying EM3DANI.')
                    mod, loge_flag = read_em3dani(modelfile)
                    dimensionality = '3d'
        elif file_format.lower() == 'modem2d':
            mod, loge_flag = read_2d(modelfile)
        elif file_format.lower() == 'em3dani':
            mod, loge_flag = read_em3dani(modelfile)
            dimensionality = '3d'
        elif file_format.lower() == 'mt3dani':
            mod, loge_flag = read_mt3dani(modelfile)
            dimensionality = '3d'
        if loge_flag:
            mod['vals'] = np.exp(mod['vals'])
        return mod, dimensionality
    except FileNotFoundError as e:
        raise(WSFileError(ID='fnf', offender=modelfile))


def read_raw_data(site_names, datpath='', edi_locs_from='definemeas', progress_bar=None):
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
                    data = data.astype(np.float64)
                    periods = data[:, 0]
                    dataReal = data[:, 1]
                    dataImag = data[:, 2]
                    dataErr = data[:, 3]
                    siteData_dict.update({''.join([comp, 'R']): dataReal})
                    siteData_dict.update({''.join([comp, 'I']): -1 * dataImag})
                    siteError_dict.update({''.join([comp, 'R']): dataErr})
                    siteError_dict.update({''.join([comp, 'I']): dataErr})
                periods[periods < 0] = -1 / periods[periods < 0]
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=file))
        try:
            assert len(set(ns_cache)) == 1
        except AssertionError:
            msg = 'Number of periods in {} is not equal for all components'.format(file)
            print('Fatal error in pyMT.IO.read_dat')
            raise(WSFileError(ID='int', offender=file, extra=msg))
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
    def read_edi(file, long_origin=999, edi_locs_from='definemeas'):
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
                      'TROT': [],
                      'FREQ': [],
                      'INFO': [],
                      '=DEFINEMEAS': []}
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
                elif in_block and ('>' in line):
                    in_block = 0
                    blocks.update({block_name: lines[block_start:ii]})
                    # print(block_name + str(ii))
                    ii -= 2
                ii += 1
            return blocks

        def read_header(header):
            lat, lon, elev = 0, 0, 0
            header_error = False
            for line in header:
                try:
                    if 'LAT' in line:
                        lat = (utils.dms2dd(line.split('=')[1].strip()))
                    if 'LONG' in line:
                        lon = (utils.dms2dd(line.split('=')[1].strip()))
                    elif 'LON' in line:  # Make exception if its spelt this way...
                        lon = (utils.dms2dd(line.split('=')[1].strip()))
                    if 'ELEV' in line:
                        elev = float(line.split('=')[1].strip())
                except ValueError:
                    header_error = True
                    lat, lon, elev = 0, 0, 0
            return lat, lon, elev, header_error

        def read_info(block):
            # print('Read info not implemented and returns only zeros')
            return 0, 0, 0

        def read_definemeas(block):
            lat, lon, elev = 0, 0, 0
            def_error = False
            # print('DEFINEMEAS')
            # print(block)
            for line in block:
                try:
                    if 'REFLAT' in line:
                        lat = (utils.dms2dd(line.split('=')[1].strip()))
                    if ('REFLONG' in line) or ('REFLON' in line):
                        lon = (utils.dms2dd(line.split('=')[1].strip()))
                    if 'REFELEV' in line:
                        elev = float(line.split('=')[1].strip())
                except ValueError:
                    def_error = True
                    lat, lon, elev = 0, 0, 0
            # print([lat, lon, elev])
            return lat, lon, elev, def_error

        def extract_location(blocks, edi_locs_from='definemeas'):
            lat_head, lon_head, elev_head, header_error = read_header(blocks['HEAD'])
            lat_info, lon_info, elev_info = read_info(blocks['INFO'])
            lat_define, lon_define, elev_define, def_error = read_definemeas(blocks['=DEFINEMEAS'])
            # if (lat_head != lat_info) or (lat_head != lat_define) or (lat_define != lat_info):
            #     print('Latitudes listed in HEAD, INFO and DEFINEMEAS do not match.')
            # if (lon_head != lon_info) or (lon_head != lon_define) or (lon_define != lon_info):
            #     print('Longitudes listed in HEAD, INFO and DEFINEMEAS do not match.')
            # if (elev_head != elev_info) or (elev_head != elev_define) or (elev_define != elev_info):
            #     print('Elevations listed in HEAD, INFO and DEFINEMEAS do not match.')
            # print('Location information extracted from DEFINEMEAS block')
            # lat, lon, elev = lat_define, lon_define, elev_define
            # print([lat, lon, elev])
            if ((lat_define == 0) or (lon_define == 0) and (edi_locs_from.lower() == 'definemeas')):
                print('Definemeas locations not set. Using Header')
                edi_locs_from = 'head'
            if ((lat_head == 0) or (lon_head == 0) and (edi_locs_from.lower() == 'head')):
                print('Header locations not set. Using Definemeas')
                edi_locs_from = 'definemeas'
            if edi_locs_from.lower() == 'definemeas':
                return lat_define, lon_define, elev_define, def_error
            elif edi_locs_from.lower() == 'head':
                return lat_head, lon_head, elev_head, header_error
            # if lat == lon == elev == 0:
                # lat, lon, elev = lat_head, lon_head, elev_head
            # print([lat, lon, elev])
            

        def read_data_block(block):
            # print(block[0])
            try:
                num_freqs = float(block[0].split('//')[1])
            except IndexError:
                print('Number of freqs not specified in data block, proceeding anyways')
                num_freqs = 0
            data = []
            for line in block[1:]:
                for f in line.split():
                    data.append(np.nan_to_num(float(f.strip())))
            if num_freqs and (len(data) != num_freqs):
                print('Number of frequencies does not match the given number')
                print('Proceeding anyways...')
            return np.array(data)

        def extract_tensor_info(blocks):
            z_azi = None # If neither TROT nor ZROT are specified, assume azi = 0
            t_azi = None # If neither TROT nor ZROT are specified, assume azi = 0
            scaling_factor = {'Z': 4 * np.pi / 10000, 'T': 1}
            for key in blocks.keys():
                # key = key.strip()
                if (key[0] == 'Z' or key[0] == 'T') and (key != 'ZROT' and key != 'TROT'):
                    if blocks[key]:
                        data_block = read_data_block(blocks[key])
                        if 'VAR' in key:
                            new_key = key[:4]
                            errors.update({new_key[:-1] + 'R': abs(data_block) * scaling_factor[key[0]]})
                            errors.update({new_key[:-1] + 'I': abs(data_block) * scaling_factor[key[0]]})
                        else:
                            if key[0] == 'Z':
                                new_key = key[:4]
                            else:
                                new_key = ''.join(['TZ', key[1:3]])
                            
                            # elif 'Z' in key:
                            data.update({new_key: data_block * scaling_factor[key[0]]})
                        # else:
                            # data.update({new_key: data_block})
                elif key == 'ZROT' and blocks[key]:
                    try:
                        data_block = read_data_block(blocks[key])
                    except IndexError:
                        print('Missing ZROT info, setting ZROT=0')
                        z_azi = 0
                        equal_rots = 1
                        continue    
                    if not np.all(data_block == data_block[1]):
                        z_azi = data_block[0]
                        equal_rots = 0
                    else:
                        z_azi = data_block[0]
                        equal_rots = 1
                elif key == 'TROT' and blocks[key]:
                    data_block = read_data_block(blocks[key])
                    if not np.all(data_block == data_block[1]):
                        t_azi = data_block[0]
                        equal_rots = 0
                    else:
                        t_azi = data_block[0]
                        equal_rots = 1
                else:
                    print('Missing ZROT info, setting ZROT=0')
                    z_azi = 0
                    equal_rots = 1
                    continue    
            if z_azi == t_azi:
                azi = z_azi
                equal_rots = 1
            else:
                if z_azi is None:
                    print('Missing ZROT info, setting to TROT')
                    azi = t_azi
                elif t_azi is None:
                    azi = z_azi
                else:
                    azi = 0
                    equal_rots = 0
            # EDI format has ZxyR, ZxyI positive; ZyxR, ZyxI negative. This needs to be changed
            # EDI is generally +iwt sign convention. Note this can be specified in EDI files, but is not read anywhere here.
            try:
                data['ZXYI'] *= -1
                data['ZYXI'] *= -1
                data['ZXXI'] *= -1
                data['ZYYI'] *= -1
            except KeyError:
                pass
            try:
                data['TZXI'] *= -1
                data['TZYI'] *= -1
            except KeyError:
                pass
            # Double check all data components have corresponding errors.
            error_flag = 0
            for component in data.keys():
                try:
                    assert(errors[component].shape == data[component].shape)
                except KeyError:
                    # print(component)
                    # print(errors.keys())
                    if component.lower().startswith('z'):
                        print('Reseting errors for {}'.format(component))
                        errors.update({component: np.abs(data[component]) * 0.05})
                    else:
                        errors.update({component: np.ones(data[component].shape) * 0.03})
                    error_flag = 1
            return data, errors, azi, error_flag, equal_rots

        data = {}
        errors = {}
        try:
            # print('Reading file: {}'.format(file))
            with open(file, 'r') as f:
                # Need to also read in INFO and DEFINEMEAS blocks to confirm that the location
                # info is consistent
                lines = f.readlines()
                blocks = extract_blocks(lines)
                Lat, Long, elev, loc_error = extract_location(blocks, edi_locs_from=edi_locs_from)
                if loc_error:
                    print('Error reading locations from {} as specified in file {}'.format(edi_locs_from, file))
                    print('Proceeding anyways')
                if blocks['FREQ']:
                    frequencies = read_data_block(blocks['FREQ'])
                else:
                    raise WSFileError(ID='int', offender=file, extra='Frequency block non-existent.')
                periods = utils.truncate(1 / frequencies)
                # if periods[0] > periods[1]:
                    # print('Periods in {} are not in ascending order'.format(file))
                data, errors, azi, error_flag, equal_rots = extract_tensor_info(blocks)
                if not equal_rots:
                    print('Not all rotations are the same, site: {}. This is not supported yet...'.format(file))
                if error_flag:
                    pass
                    # print('Errors not properly specified in {}.'.format(file))
                    # print('Setting offending component errors to floor values.')
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
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=file))

    siteData = {}
    long_origin = 999
    if os.name == 'nt':
        connector = '\\'
    else:
        connector = '/'
    if datpath:
        path = datpath + connector
    else:
        path = './'
    all_dats = [file for file in os.listdir(path) if file.endswith('.dat')]
    # all_edi = [file for file in os.listdir(path) if file.endswith('.edi')]
    for ii, site in enumerate(site_names):
        if progress_bar:
            if ii == 0:
                progress_bar.input_load.emit('raw')
            progress_bar.counter.emit()
        #     progress_bar.setLabelText('Reading Data...')
        #     progress_bar.setValue(ii)
        # Look for J-format files first
        # print(site)
        if ''.join([site, '.dat']) in all_dats:
            if site.endswith('.dat'):
                file = ''.join([path, site])
            else:
                file = ''.join([path, site, '.dat'])
            # try:
            site_dict, long_origin = read_dat(file, long_origin)
            siteData.update({site: site_dict})
            # except WSFileError:
            #     print('{} not found. Continuing without it.'.format(file))
        else:
            if site.endswith('.edi'):
                file = ''.join([path, site])
            else:
                file = ''.join([path, site, '.edi'])
            # try:
            site_dict, long_origin = read_edi(file, long_origin, edi_locs_from=edi_locs_from)
            site = site.replace('.edi', '')
            site = site.replace('.dat', '')
            siteData.update({site: site_dict})
            # # except WSFileError as e:
            #         print(e.message)
            #         print('Skipping site...')

    return siteData


def read_sites(listfile):
    """Summary

    Args:
        listfile (TYPE): Description

    Returns:
        TYPE: Description
    """
    # If the given file is just a single EDI, set the list to just that file
    if listfile.endswith('.edi'):
        return [os.path.basename(listfile).replace('.edi', '')]
    try:
        with open(listfile, 'r') as f:
            # If the list doesn't have the number of stations, just read them anyways.
            ns = next(f)
            site_names = list(filter(None, f.read().split('\n')))
            try:
                ns = int(ns)
            except ValueError:
                site_names.insert(0, ns.replace('.edi','').replace('.dat','').strip())
            site_names = [name.replace('.dat', '') for name in site_names]
            site_names = [name.replace('.edi', '') for name in site_names]

            # if ns != len(site_names):
            #     raise(WSFileError(ID='int', offender=listfile,
            #                       extra='# Sites does not match length of list.'))
        return site_names
    except FileNotFoundError:
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


def read_data_header(datafile=None, file_format=None):
    # Just read whatever is necessary to get the number of sites, periods, components from the file

    def read_ws_header(datafile):
        with open(datafile, 'r') as f:
            header = f.readline().strip().split()
            NS, NP, NC, bg = [int(x) for x in header]
        return NS, NP

    def read_modem_header(datafile):
        lines = []
        with open(datafile, 'r') as f:
            lines.append(f.readline().strip())
            while True:
                lines.append(f.readline().strip())
                if lines[-1][0] not in ('#', '>'):
                    break
        NP, NS = [int(x) for x in lines[-2].split()[1:]]
        return NS, NP

    def read_em3dani_header(datafile):
        pass

    def read_mare2dem_header(datafile):
        cc = 0
        with open(datafile, 'r') as f:
            while cc < 2:
                line = f.readline()
                if 'mt frequencies' in line.lower():
                    NP = int(line.strip().split()[-1])
                    cc += 1
                if 'mt receivers' in line.lower():
                    NS = int(line.strip().split()[-1])
                    cc += 1
        return NS, NP

    ext = os.path.splitext(datafile)[1]
    if not file_format:
        if ext == '.emdata':
            file_format = 'MARE2DEM'
        elif ext == '.dat':
            file_format = 'ModEM'
        elif ext == '.adat' or ext == '.resp':
            file_format = 'em3dani'
        elif ext == '.data' or 'resp' in datafile:
            file_format = 'WSINV3DMT'

    if file_format.lower() == 'wsinv3dmt':
        # WSINV, em3dani, and ModEM may share the same .data file extension, so try them all
        try:
            return read_ws_header(datafile)
        except ValueError:
            pass
        try:
            return read_em3dani_header(datafile)
        except IndexError:
            pass
        try:
            return read_modem_header(datafile)
        except ValueError:
            print('Output format {} not recognized'.format(file_format))
            raise WSFileError(ID='fmt', offender=file_format, expected=('mare2dem',
                                                                        'wsinv3dmt',
                                                                        'ModEM',
                                                                        'em3dani'))    
    elif file_format.lower() == 'modem':
        return read_modem_header(datafile)
    elif file_format.lower() == 'mare2dem':
        return read_mare2dem_header(datafile)
    elif file_format.lower() == 'em3dani':
        return read_em3dani_header(datafile)
    else:
        print('Output format {} not recognized'.format(file_format))
        raise WSFileError(ID='fmt', offender=file_format, expected=('mare2dem',
                                                                    'wsinv3dmt',
                                                                    'ModEM',
                                                                    'em3dani'))


def read_data(datafile='', site_names='', file_format='modem', invType=None):
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
                # Skip comment lines
                while True:
                    header = next(f)
                    if not header.strip().startswith('#'):
                        break
                NS, NP, *NR = [round(float(h), 1) for h in header.split()]
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
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        other_info = {'inversion_type': invType,
                      'site_names': site_names,
                      'UTM_zone': 'Undefined',
                      'origin': (0, 0),
                      'dimensionality': '3D'}
        return sites, other_info

    def read_modem_data(datafile='', site_names='', invType=None):
        COMPONENTS_3D = ('Full_Impedance', 'Full_Vertical_Components',
                         'Off_Diagonal_Impedance', 'Phase_Tensor')
        COMPONENTS_2D = ('TE_Impedance', 'TM_Impedance')
        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        marker = -10
        block_start = []
        periods = []
        sites = {}
        site_data = {}
        site_error = {}
        # site_errmap = {}
        site_locations = {}
        site_periods = {}
        inv_type = 0
        new_site_names = []
        all_data_types = []
        if site_names:
            site_data = {site: {} for site in site_names}
            site_error = {site: {} for site in site_names}
            site_locations = {site: {'X': [], 'Y': [], 'elev': []} for site in site_names}
            site_periods = {site: [] for site in site_names}
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
            current_data_type = lines[line_number + 2].split('>')[1].strip()
            sign_convention = lines[line_number + 3].split('>')[1].strip()
            units = lines[line_number + 4].split('>')[1].strip()
            sign_multiplier, scale_factor = 1, 1
            if '+' in sign_convention:
                sign_multiplier = -1
            if units == '[mV/km]/[nT]':
                scale_factor = 4 * np.pi / 10000
            azimuth = float(lines[line_number + 5].split('>')[1].strip())
            o_x, *o_y = [float(x) for x in lines[line_number + 6].split('>')[1].strip().split()]
            o_y = o_y[0]
            NP, NS = [int(x) for x in lines[line_number + 7].split('>')[1].strip().split()]
            all_data_types.append(current_data_type)
            if current_data_type in COMPONENTS_3D:
                dimensionality = '3D'
                for line_string in lines[block_start[ii] + 8: block_start[ii + 1]]:
                    line = line_string.split()
                    period = float(line[0])
                    # periods.append(float(line[0]))
                    site_name = line[1]
                    X, Y, Z = [float(x) for x in line[4:7]]
                    if site_name not in site_data.keys():
                        site_data.update({site_name: {}})
                        site_error.update({site_name: {}})
                        site_locations.update({site_name: {'X': [],
                                                           'Y': [],
                                                           'elev': []}})
                        site_periods.update({site_name: []})

                    if site_name not in new_site_names:
                        new_site_names.append(site_name)
                    site_locations[site_name]['X'] = X
                    site_locations[site_name]['Y'] = Y
                    site_locations[site_name]['elev'] = Z
                    if period not in site_periods[site_name]:
                        site_periods[site_name].append(period)

                    component = line[7]
                    # if inv_type <= 5:
                    if component.upper().startswith('Z') or component.upper().startswith('T'):
                        if component == 'TX' or component == 'TY':
                            component = component[0] + 'Z' + component[1]
                        real, imag = [float(x) for x in line[8:10]]
                        error = float(line[10])


                        # site_data[site_name].update({component + 'I': {period: imag}})
                        # site_error[site_name].update({component + 'R': {period: error}})
                        # site_error[site_name].update({component + 'I': {period: error}})
                        if component + 'R' not in site_data[site_name].keys():
                            site_data[site_name].update({component + 'R': {}})
                        # if component + 'I' not in site_data[site_name].keys():
                            site_data[site_name].update({component + 'I': {}})
                        if component + 'R' not in site_error[site_name].keys():
                            site_error[site_name].update({component + 'R': {}})
                        # if component + 'I' not in site_error[site_name].keys():
                            site_error[site_name].update({component + 'I': {}})
                        site_data[site_name][component + 'R'].update({period: scale_factor * real})
                        site_data[site_name][component + 'I'].update({period: sign_multiplier * scale_factor * imag})
                        site_error[site_name][component + 'R'].update({period: scale_factor * error})
                        site_error[site_name][component + 'I'].update({period: scale_factor * error})
                        # site_data[site_name][component + 'R'].append(real)
                        # site_data[site_name][component + 'I'].append(imag)
                        # site_error[site_name][component + 'R'].append(error)
                        # site_error[site_name][component + 'I'].append(error)
                    elif component.upper().startswith('PT'):
                        real = float(line[8])
                        error = float(line[9])
                        # For now we are swapping the X and Y's to conform with the Caldwell et al. definition
                        swapped_component = component.replace('X', '1')
                        swapped_component = swapped_component.replace('Y', 'X')
                        swapped_component = swapped_component.replace('1', 'Y')
                        if swapped_component not in site_data[site_name].keys():
                            site_data[site_name].update({swapped_component: {}})
                        if swapped_component not in site_error[site_name].keys():
                            site_error[site_name].update({swapped_component: {}})
                        site_data[site_name][swapped_component].update({period: real})
                        site_error[site_name][swapped_component].update({period: error})
                        # if swapped_component not in site_data[site_name].keys():
                        #     site_data[site_name].update({swapped_component: []})
                        # if swapped_component not in site_error[site_name].keys():
                        #     site_error[site_name].update({swapped_component: []})
                        # site_data[site_name][swapped_component].append(real)
                        # site_error[site_name][swapped_component].append(error)
                    # print(site_data[site_lookup[code]][component + 'R'])
                    else:
                        print('Component {} not understood. Skipping'.format(component))

            else:
                dimensionality = '2D'
                for line_string in lines[block_start[ii] + 8: block_start[ii + 1]]:
                    line = line_string.split()
                    # periods.append(float(line[0]))
                    period = float(line[0])
                    site_name = line[1]
                    X, Y, Z = [float(x) for x in line[4:7]]
                    if site_name not in site_data.keys():
                        site_data.update({site_name: {}})
                        site_error.update({site_name: {}})
                        site_locations.update({site_name: {'X': [],
                                                           'Y': [],
                                                           'elev': []}})
                        site_periods.update({site_name: []})
                    if site_name not in new_site_names:
                        new_site_names.append(site_name)
                    site_locations[site_name]['X'] = X
                    site_locations[site_name]['Y'] = Y
                    site_locations[site_name]['elev'] = Z
                    if period not in site_periods[site_name]:
                        site_periods[site_name].append(period)
                    component = line[7]
                    real, imag = [float(x) for x in line[8:10]]
                    error = float(line[10])
                    if component == 'TE':
                        if 'ZXYR' not in site_data[site_name].keys():
                            site_data[site_name].update({'ZXYR': {}})
                            site_data[site_name].update({'ZXYI': {}})
                            site_error[site_name].update({'ZXYR': {}})
                            site_error[site_name].update({'ZXYI': {}})
                        site_data[site_name]['ZXYR'].update({period: scale_factor * real})
                        site_data[site_name]['ZXYI'].update({period: scale_factor * sign_multiplier * imag})
                        site_error[site_name]['ZXYR'].update({period: scale_factor * error})
                        site_error[site_name]['ZXYI'].update({period: scale_factor * error})
                    if component == 'TM':
                        if 'ZYXR' not in site_data[site_name].keys():
                            site_data[site_name].update({'ZYXR': {}})
                            site_data[site_name].update({'ZYXI': {}})
                            site_error[site_name].update({'ZYXR': {}})
                            site_error[site_name].update({'ZYXI': {}})
                        site_data[site_name]['ZYXR'].update({period: scale_factor * real})
                        site_data[site_name]['ZYXI'].update({period: scale_factor * sign_multiplier * imag})
                        site_error[site_name]['ZYXR'].update({period: scale_factor * error})
                        site_error[site_name]['ZYXI'].update({period: scale_factor * error})
        # periods = np.unique(np.array(periods))
        # print(site_data[site_names[0]])
        # site_errmap[site][component] = np.array(site_errmap[site][component])
        # print(site_data[site][component])
        if not site_names:
            site_names = new_site_names
        elif site_names != new_site_names:
            print('Site names specified in list file do not match those in {}\n'.format(datafile))
            if set(site_names) == set(new_site_names):
                print('Looks like just the order of sites is different. Proceeding anyways, watch out for buggy behavior...')
                site_names = new_site_names
            else:
                print('Proceeding with names set in list file.\n')

        all_periods = []
        sorted_site_data = copy.deepcopy(site_data)
        sorted_site_error = copy.deepcopy(site_error)
        all_periods = np.unique(np.array(utils.flatten_list([site_periods[site] for site in new_site_names])))
        all_components = set(utils.flatten_list([list(site_data[site].keys()) for site in site_data.keys()]))
        for site in new_site_names:
            thrown_error = 0
            # idx = np.argsort(site_periods[site])
            periods = sorted(site_periods[site])
            for component in all_components:
                data, error = [], []
                if component in site_data[site].keys():
                    for period in all_periods:
                        try:
                            data.append(site_data[site][component][period])
                            error.append(site_error[site][component][period])
                        except KeyError:
                            if thrown_error == 0:
                                print('Non-uniform period map at site {}. Creating dummy data.'.format(site))
                                thrown_error = 1
                            idx = np.argmin(abs(period - periods))
                            if periods[idx] not in site_data[site][component].keys():
                                if thrown_error == 1:
                                    print('Non-uniform component usage at site {}. Creating even dummier data.'.format(site))
                                    thrown_error = 2
                                idx = list(site_data[site][component].keys())[0]
                                data.append(site_data[site][component][idx])
                            else:
                                data.append(site_data[site][component][periods[idx]])
                            error.append(REMOVE_FLAG)
                else:
                    data = np.zeros((all_periods.shape))
                    error = np.ones((all_periods.shape)) * REMOVE_FLAG
                sorted_site_data[site].update({component: np.array(data)})
                sorted_site_error[site].update({component: np.array(error)})
                # sorted_periods = sorted(site_data[site][component].keys())
                # sorted_site_data[site].update({component: np.array([site_data[site][component][period] for period in sorted_periods])})
                # sorted_site_error[site].update({component: np.array([site_error[site][component][period] for period in sorted_periods])})
        # Sites get named according to list file, but are here internally called
        try:
            inv_type = [key for key in INVERSION_TYPES.keys() if all_components == set(INVERSION_TYPES[key])][0]
        except IndexError:
            msg = 'Components listed in {} are not yet a supported inversion type. Sorry ¯\\_(-_-)_/¯'.format(datafile)
            raise(WSFileError(ID='int', offender=datafile, extra=msg))
        # debug_print(inv_type, 'debug.log')
        # according to the data file
        for ii, site in enumerate(site_names):

            sites.update({site: {
                          'data': sorted_site_data[new_site_names[ii]],
                          'errors': sorted_site_error[new_site_names[ii]],
                          'periods': all_periods,
                          'locations': site_locations[new_site_names[ii]],
                          'azimuth': azimuth,
                          'errFloorZ': 0,
                          'errFloorT': 0}
                          })
        other_info = {'inversion_type': inv_type,
                      'site_names': site_names,
                      'origin': (o_x, o_y),
                      'UTM_zone': UTM_zone,
                      'dimensionality': dimensionality}
        return sites, other_info

    def read_modem_data_old(datafile='', site_names='', invType=None):
        #  Will only ready Impedance and TF data so far, not rho/phase
        # print('Inside read_modem_data')
        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        marker = -10
        block_start = []
        periods = []
        sites = {}
        site_data = {}
        site_error = {}
        # site_errmap = {}
        site_locations = {}
        site_periods = {}
        inv_type = 0
        new_site_names = []
        if site_names:
            site_data = {site: {} for site in site_names}
            site_error = {site: {} for site in site_names}
            site_locations = {site: {'X': [], 'Y': [], 'elev': []} for site in site_names}
            site_periods = {site: [] for site in site_names}
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
                dimensionality = '3D'
                if inv_type == 3:
                    inv_type = 5
                else:
                    inv_type = 1
            #     components = ('ZXX', 'ZYY', 'ZXY', 'ZYR')
            elif data_type == 'Full_Vertical_Components':
                dimensionality = '3D'
                if inv_type == 1:
                    inv_type = 5
                elif inv_type == 2:
                    inv_type = 4
                elif inv_type == 6:
                    inv_type = 7
                else:
                    inv_type = 3
            #     components = ('TX', 'TY')
            elif data_type == 'Off_Diagonal_Impedance':
                dimensionality = '3D'
                if inv_type == 3:
                    inv_type = 4
                else:
                    inv_type = 2
            #     # components = ('ZXY', 'ZYX')
            elif data_type == 'Phase_Tensor':
                dimensionality = '3D'
                if inv_type == 3:
                    inv_type = 7
                else:
                    inv_type = 6
            elif data_type == 'TE_Impedance':
                dimensionality = '2D'
                if inv_type == 9:
                    inv_type = 10
                else:
                    inv_type = 8
            elif data_type == 'TM_Impedance':
                dimensionality = '2D'
                if inv_type == 8:
                    inv_type = 10
                else:
                    inv_type = 9
            if dimensionality == '3D':
                for line_string in lines[block_start[ii] + 8: block_start[ii + 1]]:
                    line = line_string.split()
                    period = float(line[0])
                    # periods.append(float(line[0]))
                    site_name = line[1]
                    X, Y, Z = [float(x) for x in line[4:7]]
                    if site_name not in site_data.keys():
                        site_data.update({site_name: {}})
                        site_error.update({site_name: {}})
                        site_locations.update({site_name: {'X': [],
                                                           'Y': [],
                                                           'elev': []}})
                        site_periods.update({site_name: []})

                    if site_name not in new_site_names:
                        new_site_names.append(site_name)
                    site_locations[site_name]['X'] = X
                    site_locations[site_name]['Y'] = Y
                    site_locations[site_name]['elev'] = Z
                    if period not in site_periods[site_name]:
                        site_periods[site_name].append(period)

                    component = line[7]
                    # if inv_type <= 5:
                    if component.upper().startswith('Z') or component.upper().startswith('T'):
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
                    elif component.upper().startswith('PT'):
                        real = float(line[8])
                        error = float(line[9])
                        # For now we are swapping the X and Y's to conform with the Caldwell et al. definition
                        swapped_component = component.replace('X', '1')
                        swapped_component = swapped_component.replace('Y', 'X')
                        swapped_component = swapped_component.replace('1', 'Y')
                        if swapped_component not in site_data[site_name].keys():
                            site_data[site_name].update({swapped_component: []})
                        if swapped_component not in site_error[site_name].keys():
                            site_error[site_name].update({swapped_component: []})
                        site_data[site_name][swapped_component].append(real)
                        site_error[site_name][swapped_component].append(error)
                    # print(site_data[site_lookup[code]][component + 'R'])
                    else:
                        print('Component {} not understood. Skipping'.format(component))

            elif dimensionality == '2D':
                for line_string in lines[block_start[ii] + 8: block_start[ii + 1]]:
                    line = line_string.split()
                    periods.append(float(line[0]))
                    site_name = line[1]
                    X, Y, Z = [float(x) for x in line[4:7]]
                    if site_name not in site_data.keys():
                        site_data.update({site_name: {}})
                        site_error.update({site_name: {}})
                        site_locations.update({site_name: {'X': [],
                                                           'Y': [],
                                                           'elev': []}})
                    if site_name not in new_site_names:
                        new_site_names.append(site_name)
                    site_locations[site_name]['X'] = X
                    site_locations[site_name]['Y'] = Y
                    site_locations[site_name]['elev'] = Z

                    component = line[7]
                    real, imag = [float(x) for x in line[8:10]]
                    error = float(line[10])
                    if component == 'TE':
                        if 'ZXYR' not in site_data[site_name].keys():
                            site_data[site_name].update({'ZXYR': []})
                            site_data[site_name].update({'ZXYI': []})
                            site_error[site_name].update({'ZXYR': []})
                            site_error[site_name].update({'ZXYI': []})
                        site_data[site_name]['ZXYR'].append(real)
                        site_data[site_name]['ZXYI'].append(imag)
                        site_error[site_name]['ZXYR'].append(error)
                        site_error[site_name]['ZXYI'].append(error)
                    if component == 'TM':
                        if 'ZYXR' not in site_data[site_name].keys():
                            site_data[site_name].update({'ZYXR': []})
                            site_data[site_name].update({'ZYXI': []})
                            site_error[site_name].update({'ZYXR': []})
                            site_error[site_name].update({'ZYXI': []})
                        site_data[site_name]['ZYXR'].append(real)
                        site_data[site_name]['ZYXI'].append(imag)
                        site_error[site_name]['ZYXR'].append(error)
                        site_error[site_name]['ZYXI'].append(error)
        # periods = np.unique(np.array(periods))
        # print(site_data[site_names[0]])
        # site_errmap[site][component] = np.array(site_errmap[site][component])
        # print(site_data[site][component])
        if not site_names:
            site_names = new_site_names
        elif site_names != new_site_names:
            print('Site names specified in list file do not match those in {}\n'.format(datafile))
            print('Proceeding with names set in list file.\n')
            # site_names = new_site_names
        all_periods = []
        for site in new_site_names:
            idx = np.argsort(site_periods[site])
            periods = sorted(site_periods[site])
            all_periods.append(periods)
            for component in site_data[site].keys():
                vals = site_data[site][component]
                site_data[site][component] = np.array(vals)[idx]
                vals = site_error[site][component]
                site_error[site][component] = np.array(vals)[idx]
        # Sites get named according to list file, but are here internally called
        all_periods = np.unique(np.array(periods))
        # according to the data file
        for ii, site in enumerate(site_names):

            sites.update({site: {
                          'data': site_data[new_site_names[ii]],
                          'errors': site_error[new_site_names[ii]],
                          'periods': all_periods,
                          'locations': site_locations[new_site_names[ii]],
                          'azimuth': azimuth,
                          'errFloorZ': 0,
                          'errFloorT': 0}
                          })
        other_info = {'inversion_type': inv_type,
                      'site_names': site_names,
                      'origin': (o_x, o_y),
                      'UTM_zone': UTM_zone,
                      'dimensionality': dimensionality}
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
                            134: 'TZYI',
                            151: 'EXR',
                            152: 'EXI',
                            153: 'EYR',
                            154: 'EYI',
                            161: 'HXR',
                            162: 'HXI',
                            163: 'HYR',
                            164: 'HYI',
                            165: 'HZR',
                            166: 'HZI'}
        fields_data_names = ('EXR',
                             'EXI',
                             'EYR',
                             'EYI',
                             'HXR',
                             'HXI',
                             'HYR',
                             'HYI',
                             'HZR',
                             'HZI')
        is_fields = False
        #  Assumes you're only working with MT data
        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        #  Remove comment lines
        lines = [line for line in lines if not line.startswith('%') and not line.startswith('!')]
        em_format = lines[0].split(':')[1].strip()
        if 'emresp' in em_format.lower():
            em_format = 'Response'
        else:
            em_format = 'Data'
        #  It's not entirely clear how strike is used in the program
        # UTM_zone, 
        UTM_zone, o_x, o_y, strike = re.split(r'\s{2,}', lines[1].split(':')[1].strip())
        # a = re.split(r'\s{2,}', lines[1].split(':')[1].strip())
        # debug_print(a, 'io.txt')
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
            try:
                site_aliases.update({ii + 1: site['Name']})
            except KeyError:
                site['Name'] = str(ii)
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
            data_type_int = int(site['Type'])
            data_type = data_type_lookup[int(site['Type'])]
            if data_type_int > 150:
                is_fields = True
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
        if is_fields: # Hacky but whatever
            fields_data = {site: {field: [] for field in fields_data_names} for site in site_dict.keys()}
            for site in site_dict.keys():
                ZXX = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                ZYY = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                ZXY = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                ZYX = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                TZX = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                TZY = np.zeros(len(site_dict[site]['data']['EXR']), dtype=np.complex128)
                for ii in range(len(site_dict[site]['data']['EXR'])):
                    # E = np.array((site_dict[site]['data']['EXR'][ii] + 1j*site_dict[site]['data']['EXI'][ii],
                                  # site_dict[site]['data']['EYR'][ii] + 1j*site_dict[site]['data']['EYI'][ii]))
                    # H = np.array((site_dict[site]['data']['HXR'][ii] + 1j*site_dict[site]['data']['HXI'][ii], 
                                  # site_dict[site]['data']['HYR'][ii] + 1j*site_dict[site]['data']['HYI'][ii]))
                    EX = np.array((site_dict[site]['data']['EXR'][ii] + 1j*site_dict[site]['data']['EXI'][ii]))
                    EY = np.array((site_dict[site]['data']['EYR'][ii] + 1j*site_dict[site]['data']['EYI'][ii]))
                    HX = np.array((site_dict[site]['data']['HXR'][ii] + 1j*site_dict[site]['data']['HXI'][ii]))
                    HY = np.array((site_dict[site]['data']['HYR'][ii] + 1j*site_dict[site]['data']['HYI'][ii]))
                    HZ = np.array((site_dict[site]['data']['HZR'][ii] + 1j*site_dict[site]['data']['HZI'][ii]))

                    # try:
                    #     Z = np.inner(np.outer(E, np.conj(H)), np.linalg.inv((np.outer(H, np.conj(H)))))
                    # except np.linalg.LinAlgError:
                    #     print('Singular matrix at site {}, assuming 2D'.format(site))
                    Z = np.zeros((2, 2), dtype=np.complex128)
                    # Z[0, 0] = (EX * np.conj(HX)) / (HX * np.conj(HX))
                    Z[0, 1] = (EX * np.conj(HY)) / (HY * np.conj(HY))
                    Z[1, 0] = (EY * np.conj(HX)) / (HX * np.conj(HX))
                    # Z[1, 1] = (EY * np.conj(HY)) / (HY * np.conj(HY))
                    Z[np.isnan(Z)] = 1e-10
                    Z[Z == 0] = 1e-10
                    #     H[H == 0] = 1e12
                    #     Z[0, 1] = E[0] / H[1]
                    #     Z[1, 0] = E[1] / H[0]

                    ZXX[ii], ZXY[ii], ZYX[ii], ZYY[ii] = Z[0, 0], Z[0, 1], Z[1, 0], Z[1, 1]
                    # TZX[ii] = 0 + 0j
                    TZY[ii] = HZ / HY
                site_dict[site]['data'].update({'ZXXR': np.real(ZXX),
                                                'ZXYR': np.real(ZXY),
                                                'ZYXR': np.real(ZYX),
                                                'ZYYR': np.real(ZYY),
                                                'TZXR': np.real(TZX),
                                                'TZYR': np.real(TZY),
                                                'ZXXI': -np.imag(ZXX),
                                                'ZXYI': -np.imag(ZXY),
                                                'ZYXI': -np.imag(ZYX),
                                                'ZYYI': -np.imag(ZYY),
                                                'TZXI': -np.imag(TZX),
                                                'TZYI': -np.imag(TZY)})
                site_dict[site]['errors'].update({'ZXXR': abs(np.real(ZXY) * 0.05),
                                                  'ZXYR': abs(np.real(ZXY) * 0.05),
                                                  'ZYXR': abs(np.real(ZYX) * 0.05),
                                                  'ZYYR': abs(np.real(ZYX) * 0.05),
                                                  'ZXXI': abs(np.imag(ZXY) * 0.05),
                                                  'ZXYI': abs(np.imag(ZXY) * 0.05),
                                                  'ZYXI': abs(np.imag(ZYX) * 0.05),
                                                  'ZYYI': abs(np.imag(ZYX) * 0.05),
                                                  'TZXR': 0.05 * np.ones(TZY.shape),
                                                  'TZYR': 0.05 * np.ones(TZY.shape),
                                                  'TZXI': 0.05 * np.ones(TZY.shape),
                                                  'TZYI': 0.05 * np.ones(TZY.shape)})
                for field in fields_data_names:
                    fields_data[site][field] = site_dict[site]['data'][field]
                    del site_dict[site]['data'][field]
                    del site_dict[site]['errors'][field]
                site_dict[site].update({'fields': fields_data[site]})
        else:
            fields = None
        other_info = {'site_names': site_names,
                      'inversion_type': None,
                      'origin': (o_x, o_y),
                      'UTM_zone': UTM_zone,
                      'dimensionality': '2d'}
        return site_dict, other_info

    def read_em3dani(datafile='', site_names=''):
        def read_data_format(lines, data, errors, site_names, components):
            for line in lines:
                freqno, rxno, comp_no, real_val, imag_val, error_val = line.split()
                freqno, rxno, comp_no = int(freqno), int(rxno), int(comp_no)
                data[site_names[rxno-1]][components[(comp_no-1)*2]][freqno-1] = float(real_val)
                data[site_names[rxno-1]][components[(comp_no-1)*2+1]][freqno-1] = float(imag_val)
                errors[site_names[rxno-1]][components[(comp_no-1)*2]][freqno-1] = float(error_val)
                errors[site_names[rxno-1]][components[(comp_no-1)*2+1]][freqno-1] = float(error_val)
                # for ii, val in enumerate(values):
                #     data[site_names[rxno-1]][components[ii]][freqno-1] = float(val)
                #     errors[site_names[rxno-1]][components[ii][freqno-1]] = float(error_val)
            return data, errors

        def read_response_format(lines, data, site_names, components):
            for line in lines:
                freqno, rxno, *values = line.split()
                freqno, rxno = int(freqno), int(rxno)
                for ii, val in enumerate(values):
                    data[site_names[rxno-1]][components[ii]][freqno-1] = float(val)
            return data

        try:
            with open(datafile, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=datafile)) from None
        header = ['#']
        counter = 0
        # Pull out the commented lines.
        while header[-1].lower().strip().startswith('#'):
            header.append(lines.pop(0))
        # Look to see if it has the file format (data or resp file)
        file_format = [string for string in header if 'format' in string.lower()][0]
        if 'data' in file_format.lower():
            file_format = 'data'
        elif 'resp' in file_format.lower():
            file_format = 'response'
        else:
            print('File format {} not recognized. Aborting.'.format(file_format))
            raise(WSFileError(ID='fmt', offender=datafile, expected='MT3DResp or MT3DData'))
        phase_convention = header[-1].split(':')[1].strip()
        NS = int(lines.pop(0).split(':')[1].strip())
        header = lines.pop(0) # Should be commented X Y Z line
        if not site_names:
            site_names = ['site_{}'.format(ii) for ii in range(NS)]
        locations = {site: {'X': [], 'Y': [], 'elev': []} for site in site_names}
        for site in site_names:
            X, Y, Z = [float(val) for val in lines.pop(0).split()]
            locations[site]['X'] = X
            locations[site]['Y'] = Y
            locations[site]['elev'] = Z
        NP = int(lines.pop(0).split(':')[1])
        periods = np.zeros((NP))
        for ii in range(NP):
            periods[ii] = 1 / float(lines.pop(0))
        data_type = lines.pop(0).split(':')[1].strip()
        data_components = ['ZXXR', 'ZXXI',
                           'ZXYR', 'ZXYI',
                           'ZYXR', 'ZYXI',
                           'ZYYR', 'ZYYI']
        inv_type = 1
        if 'tipper' in data_type.lower():
            data_components += ['TZXR', 'TZXI', 'TZYR', 'TZYI']
            inv_type = 5
        NR = int(lines.pop(0).split(':')[1])
        if NR != len(data_components) / 2:
            print('Number of components does not match the listed data type, proceeding anyways.')
        for ii in range(NR):
            lines.pop(0)
        num_data = int(lines.pop(0).split(':')[1])
        header = lines.pop(0)
        data = {site: {comp: np.zeros((NP)) for comp in data_components} for site in site_names}
        errors = {site: {comp: np.ones((NP)) for comp in data_components} for site in site_names}
        if file_format == 'data':
            data, errors = read_data_format(lines, data, errors, site_names, data_components)
        else:
            data = read_response_format(lines, data, site_names, data_components)

        sites = {}
        for site in site_names:
            sites.update({site: {
                  'data': data[site],
                  'errors': errors[site],
                  'periods': periods,
                  'locations': locations[site],
                  'azimuth': 0,
                  'errFloorZ': 0,
                  'errFloorT': 0}
                  })
        other_info = {'inversion_type': inv_type,
                      'site_names': site_names,
                      'origin': (0, 0),
                      'UTM_zone': None,
                      'dimensionality': '3d'}
        return sites, other_info

    def read_gofem_locations(receiver_file='', backup_path=None):
        def read_file(receiver_file):
            with open(receiver_file, 'r') as f:
                lines = f.readlines()
            return lines
        read_attempt = 0
        while True:
            try:
                lines = read_file(receiver_file)
                break
            except (FileNotFoundError, TypeError):
                if read_attempt == 0:
                    receiver_file = PATH_CONNECTOR.join([backup_path, 'receivers.csv'])
                elif read_attempt == 1:
                    receiver_file = PATH_CONNECTOR.join([backup_path, '..', 'receivers.csv'])
                else:
                    raise(WSFileError(ID='fnf', offender=receiver_file,
                          extra='GoFEM Receiver file not found')) from None
                read_attempt += 1
        locations = {}
        for line in lines:
            dipole, site_name, num, X, Y, Z = line.split()
            locations.update({site_name: {'Y': float(X),
                                          'X': float(Y),
                                          'elev': float(Z)}})
        return locations


    def read_gofem(data_file, receiver_file=None):

        # This will need to be updated to allow phase tensor inversions later
        comp_dict = {'RealZxx': 'ZXXR',
                     'ImagZxx': 'ZXXI',
                     'RealZxy': 'ZXYR',
                     'ImagZxy': 'ZXYI',
                     'RealZyx': 'ZYXR',
                     'ImagZyx': 'ZYXI',
                     'RealZyy': 'ZYYR',
                     'ImagZyy': 'ZYYI',
                     'RealTzx': 'TZXR',
                     'ImagTzx': 'TZXI',
                     'RealTzy': 'TZYR',
                     'ImagTzy': 'TZYI'}
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise(WSFileError(ID='fnf', offender=data_file)) from None
        site_names = []
        data = {}
        errors = {}
        freq_order = {}
        all_freqs = []
        used_comps = []
        for line in lines:
            if line[0] == '#':
                continue
            go_comp, freq, plane, site_name, val, error_val = [x.strip() for x in line.split()]
            comp = comp_dict[go_comp]
            used_comps.append(comp)
            if site_name not in site_names:
                site_names.append(site_name)
                data.update({site_name: {comp : [] for comp in comp_dict.values()}})
                errors.update({site_name: {comp : [] for comp in comp_dict.values()}})
                freq_order.update({site_name: {comp : [] for comp in comp_dict.values()}})
            all_freqs.append(float(freq))
            data[site_name][comp_dict[go_comp]].append(float(val))
            errors[site_name][comp_dict[go_comp]].append(float(error_val))
            freq_order[site_name][comp_dict[go_comp]].append(float(freq))
        # debug_print(all_freqs, 'debug.txt')
        all_freqs = set(all_freqs)
        # debug_print(all_freqs, 'debug.txt')
        used_comps = set(used_comps)
        # Make sure all the site data is set up properly
        for site in site_names:
            for comp in used_comps:
                # Reset data to -iwt time dependence
                if comp.endswith('I'):
                    data[site][comp] = [-1*x for x in data[site][comp]]
                # Check that the site has all required frequencies
                if set(freq_order[site][comp]) != all_freqs:
                    for freq in all_freqs:
                        if freq not in freq_order[site][comp]:
                            print('Infilling frequency {} at {}, {}'.format(freq, site, comp))
                            try:
                                idx = np.argmin(abs(freq - np.array(freq_order[site][comp])))
                                data[site][comp].append(data[site][comp][idx])
                            except ValueError:
                                data[site][comp].append(0.00001)
                            errors[site][comp].append(REMOVE_FLAG)
                            freq_order[site][comp].append(freq)
                idx = np.argsort(freq_order[site][comp])
                # Reorder the points (and reverse them to get them in period order)
                data[site][comp] = np.array(data[site][comp])[idx][::-1]
                errors[site][comp] = np.array(errors[site][comp])[idx][::-1]
                freq_order[site][comp] = np.array(freq_order[site][comp])[idx][::-1]
            unused_comps = [x for x in comp_dict.values() if x not in used_comps]
            for comp in unused_comps:
                del data[site][comp]
                del errors[site][comp]
                del freq_order[site][comp]
        
        periods = np.sort(np.array([1/x for x in list(all_freqs)]))
        # debug_print(periods, 'debug.txt')
        try:
            inv_type = [key for key in INVERSION_TYPES.keys() if set(used_comps) == set(INVERSION_TYPES[key])][0]
        except IndexError:
            msg = 'Components listed in {} are not yet a supported inversion type. Sorry ¯\\_(-_-)_/¯'.format(datafile)
            raise(WSFileError(ID='int', offender=data_file, extra=msg))

        sites = {}
        backup_path = os.path.split(os.path.abspath(data_file))[0]
        locations = read_gofem_locations(receiver_file=receiver_file, backup_path=backup_path)
        for site in site_names:
            sites.update({site: {
                  'data': data[site],
                  'errors': errors[site],
                  'periods': periods,
                  'locations': locations[site],
                  'azimuth': 0,
                  'errFloorZ': 0,
                  'errFloorT': 0}
                  })
        other_info = {'inversion_type': inv_type,
                      'site_names': site_names,
                      'origin': (0, 0),
                      'UTM_zone': None,
                      'dimensionality': '3d'}
        return sites, other_info

    if file_format.lower() == 'wsinv3dmt':
        # WSINV, em3dani, and ModEM may share the same .data file extension, so try them all
        try:
            return read_ws_data(datafile, site_names, invType)
        except ValueError:
            pass
        try:
            return read_em3dani(datafile, site_names)
        except IndexError:
            pass
        try:
            return read_modem_data(datafile, site_names, invType)
        except ValueError:
            print('Output format {} not recognized'.format(file_format))
            raise WSFileError(ID='fmt', offender=file_format, expected=('mare2dem',
                                                                        'wsinv3dmt',
                                                                        'ModEM',
                                                                        'em3dani'))    
    elif file_format.lower() == 'modem':
        return read_modem_data(datafile, site_names, invType)
    elif file_format.lower() == 'mare2dem':
        return read_mare2dem_data(datafile, site_names, invType)
    elif file_format.lower() == 'em3dani':
        return read_em3dani(datafile, site_names)
    elif file_format.lower() == 'gofem':
        return read_gofem(datafile, site_names)
    else:
        print('Output format {} not recognized'.format(file_format))
        raise WSFileError(ID='fmt', offender=file_format, expected=('mare2dem',
                                                                    'wsinv3dmt',
                                                                    'ModEM',
                                                                    'em3dani'))


def write_locations(data, out_file=None, file_format='csv', verbose=0):
    def write_shapefile(data, outfile):
        if not outfile.endswith('.shp'):
            outfile += '.shp'
        if data.site_names:
            print('Writing shapefile with locations stored in data.locations')
            w = shapefile.Writer(shapefile.POINT)
            w.field('X', 'F', 10, 5)
            w.field('Y', 'F', 10, 5)
            w.field('Z', 'F', 10, 5)
            w.field('Label')
            for ii, site in enumerate(data.site_names):
                X, Y, Z = (data.locations[ii, 1],
                           data.locations[ii, 0],
                           data.sites[site].locations['elev'])
                w.point(X, Y, Z)
                w.record(X, Y, Z, site)
            w.save(outfile)

    def write_csv(data, outfile):
        with open(outfile, 'w') as f:
            if verbose:
                f.write('Station, X, Y, Elevation, Latitude, Longitude\n')
            else:
                f.write('Station, X, Y, Elevation\n')
            for ii, site in enumerate(data.site_names):
                # f.write('{:>6s}, {:>12.8g}, {:>12.8g}, {:>12.8g}\n'.format(site,
                #                                                            data.locations[ii, 1],
                #                                                            data.locations[ii, 0],
                #                                                            data.sites[site].locations['elev']))
                string = '{:>6s}, {:>12.8g}, {:>12.8g}, {:>12.8g}'.format(site,
                                                                          data.locations[ii, 1],
                                                                          data.locations[ii, 0],
                                                                          data.sites[site].locations['elev'])
                if verbose:
                    string += ', {:>8.5g}, {:>8.5g}'.format(data.sites[site].locations['Lat'],
                                                            data.sites[site].locations['Long'])
                string += '\n'
                f.write(string)

    if file_format.lower() not in ('csv', 'shp', 'kml'):
        print('File format {} not supported'.format(file_format))
        return
    if file_format.lower() == 'csv':
        write_csv(data, out_file)
    elif file_format.lower() == 'shp':
        write_shapefile(data, out_file)
    elif file_format.lower() == 'kml':
        print('Not implemented yet')


def sites_to_vtk(data, origin=None, outfile=None, UTM=None, sea_level=0, use_elevation=False):
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
    if use_elevation:
        zlocs = sea_level - np.array([data.sites[site].locations['elev'] for site in data.site_names])
    else:
        zlocs = sea_level * np.ones(xlocs.shape)
    ns = len(xlocs)
    with open(outfile, 'w') as f:
        f.write(version)
        f.write('UTM: {} \n'.format(UTM))
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        # f.write('DIMENSIONS {} {} {} \n'.format(ns, ns, 1))
        f.write('POINTS {} float\n'.format(ns))
        for ix, iy, iz in zip(xlocs, ylocs, zlocs):
            f.write('{} {} {}\n'.format(ix, iy, iz))
        f.write('POINT_DATA {}\n'.format(ns))
        f.write('SCALARS dummy float\n')
        f.write('LOOKUP_TABLE default\n')
        for ii in range(ns):
            f.write('{}\n'.format(999999))


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

def write_edi(site, out_file, info=None, header=None, mtsect=None, defs=None):
    freqs = [round(1/x, 8) for x in site.periods]
    NP = len(freqs)
    scale_factor = 1 / (4 * np.pi / 10000)
    lat_deg, lat_min, lat_sec = utils.dd2dms(site.locations['Lat'])
    long_deg, long_min, long_sec = utils.dd2dms(site.locations['Long'])
    default_header = OrderedDict([('ACQBY', '"eroots"'),
                                  ('FILEBY',   '"pyMT"'),
                                  ('FILEDATE', datetime.datetime.today().strftime('%m/%d/%y')),
                                  ('LAT', '{:d}:{:d}:{:4.2f}'.format(int(lat_deg), int(lat_min), lat_sec)),
                                  ('LONG', '{:d}:{:d}:{:4.2f}'.format(int(long_deg), int(long_min), long_sec)),
                                  ('ELEV', 0),
                                  ('STDVERS', '"SEG 1.0"'),
                                  # ('PROGVERS', '"ztem2edi {}"'.format(pkg_resources.get_distribution('ztem2edi').version)),
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

        f.write('>FREQ //{}\n'.format(NP))
        for freq in freqs:
            f.write('{:>14.4E}'.format(freq))
        f.write('\n\n')

        if set(site.IMPEDANCE_COMPONENTS).issubset(set(site.components)):
            f.write('>ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>14.3f}'.format(0))
            f.write('\n\n')

            f.write('>ZXXR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.data['ZXXR'][ii]))
            f.write('\n\n')

            f.write('>ZXXI ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * scale_factor * site.data['ZXXI'][ii]))
            f.write('\n\n')

            f.write('>ZXX.VAR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.errors['ZXXR'][ii]))
            f.write('\n\n')

            f.write('>ZYYR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.data['ZYYR'][ii]))
            f.write('\n\n')

            f.write('>ZYYI ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * scale_factor * site.data['ZYYI'][ii]))
            f.write('\n\n')

            f.write('>ZYY.VAR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.errors['ZYYR'][ii]))
            f.write('\n\n')

            f.write('>ZXYR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.data['ZXYR'][ii]))
            f.write('\n\n')

            f.write('>ZXYI ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * scale_factor * site.data['ZXYI'][ii]))
            f.write('\n\n')

            f.write('>ZXY.VAR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.errors['ZXYR'][ii]))
            f.write('\n\n')

            f.write('>ZYXR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.data['ZYXR'][ii]))
            f.write('\n\n')

            f.write('>ZYXI ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * scale_factor * site.data['ZYXI'][ii]))
            f.write('\n\n')

            f.write('>ZYX.VAR ROT=ZROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(scale_factor * site.errors['ZYXR'][ii]))
            f.write('\n\n')

        if set(site.TIPPER_COMPONENTS).issubset(set(site.components)):
            f.write('>TROT //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>14.3f}'.format(0))
            f.write('\n\n')

            f.write('>TXR.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(site.data['TZXR'][ii]))
            f.write('\n\n')

            f.write('>TXI.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * site.data['TZXI'][ii]))
            f.write('\n\n')

            f.write('>TYR.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(site.data['TZYR'][ii]))
            f.write('\n\n')

            f.write('>TYI.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(-1 * site.data['TZYI'][ii]))
            f.write('\n\n')

            f.write('>TXVAR.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(site.errors['TZXR']))
            f.write('\n\n')
            f.write('>TYVAR.EXP //{}\n'.format(NP))
            for ii in range(NP):
                f.write('{:>18.7E}'.format(site.errors['TZYR']))
            f.write('\n\n')

        f.write('>END')


def write_covariance(file_name, NX, NY, NZ, exceptions=None, sigma_x=0.3, sigma_y=0.3, sigma_z=0.3, num_smooth=1):
    header = ['+', '|', '|', '|', '|', '|', '|', '|', '|',
              '|', '|', '|', '|', '|', '|', '+']

    def write_smooth_block(file, N, sigma):
        if not isinstance(sigma, list):
            file.write(' '.join([str(sigma)] * N))
        elif len(sigma) == N:
            for sig in sigma:
                f.write('{} '.format(sig))
        else:
            print('Length of sigma not equal to mesh size: length(sigma) = {}, N = {}'.format(len(sigma), N))
            print('Printing default covariance instead.')
            sigma = sigma[0]
            file.write(' '.join([sigma] * N))
        file.write('\n')
        file.write('\n')

    def write_exceptions_block(f, expections):
        f.write('0')
        f.write('\n')
        nx, ny, nz = exceptions.shape
        for iz in range(nz):
            f.write('\n{} {}\n'.format(iz + 1, iz + 1))
            first_z = 1
            for ix in range(nx):
                if not first_z:
                    f.write('\n')
                first_z = 0
                for iy in range(ny):
                    f.write('{} '.format(int(exceptions[nx - ix - 1, iy, iz])))

    if not (file_name.endswith('.cov')):
        file_name += '.cov'
    with open(file_name, 'w') as f:
        f.write('\n'.join(header))
        f.write('\n')
        f.write('\n{} {} {}\n\n'.format(NX, NY, NZ))
        write_smooth_block(f, NZ, sigma_x)
        write_smooth_block(f, NZ, sigma_y)
        f.write('{}\n\n'.format(sigma_z))
        f.write('{}\n\n'.format(num_smooth))
        if exceptions is not None:
            write_exceptions_block(f, exceptions)
        else:
            f.write('0')


def write_data(data, outfile=None, to_write=None, file_format='ModEM', use_elevation=False, include_flagged=True):
    #  Writes out the contents of a Data object into format specified by 'file_format'
    #  Currently implemented options include WSINV3DMT and ModEM3D.
    #  Plans to implement OCCAM2D, MARE2DEM, and ModEM2D.
    def write_em3dani(data, outfile):
        if not outfile.lower().endswith('.adat'):
            outfile += '.adat'
        with open(outfile, 'w') as f:
            f.write('# Format:       MT3DData_1.0\n')
            f.write('# Description:  written by pyMT\n')
            f.write('Phase Convention:  lag\n')
            f.write('Receiver Location (m):    {}\n'.format(data.NS))
            f.write('#         X          Y           Z\n')
            for ii in range(data.NS):
                f.write('{:<14.2f} {:<14.2f} {:<14.2f}\n'.format(data.locations[ii, 0],
                                                                 data.locations[ii, 1],
                                                                 0)) # No topography in this code yet?
            f.write('Frequencies (Hz):     {}\n'.format(data.NP))
            for ii in range(data.NP):
                f.write('{:<10.5e}\n'.format(1 / data.periods[ii]))
            if data.inv_type == 1:
                f.write('DataType:   Impedance\n')
                NR = 4
            elif data.inv_type == 5:
                f.write('DataType:   Impedance_Tipper\n')
                NR = 6
            else:
                print('Inversion Type {} not allowed for em3dani. Writing Impedances only.'.format(data.inv_type))
                f.write('DataType:   Impedance\n')
            f.write('DataComp:    {}\n'.format(NR))
            f.write('ZXX\nZXY\nZYX\nZYY\n')
            if NR == 6:
                f.write('TZX\nTZY\n')
            f.write('Data Block: {}\n'.format(int(data.NP * data.NS * NR)))
            f.write('# FreqNo.  RxNo.    DCompNo.     RealValue      ImagValue      Error\n')
            for ii, site in enumerate(data.site_names):
                for jj in range(NR):
                    if jj == 0:
                        vals = data.sites[site].data['ZXXR'] + 1j*data.sites[site].data['ZXXI']
                        errors = np.abs(data.sites[site].used_error['ZXXR'] + 1j*data.sites[site].used_error['ZXXI'])
                    if jj == 1:
                        vals = data.sites[site].data['ZXYR'] + 1j*data.sites[site].data['ZXYI']
                        errors = np.abs(data.sites[site].used_error['ZXYR'] + 1j*data.sites[site].used_error['ZXYI'])
                    if jj == 2:
                        vals = data.sites[site].data['ZYXR'] + 1j*data.sites[site].data['ZYXI']
                        errors = np.abs(data.sites[site].used_error['ZYXR'] + 1j*data.sites[site].used_error['ZYXI'])
                    if jj == 3:
                        vals = data.sites[site].data['ZYYR'] + 1j*data.sites[site].data['ZYYI']
                        errors = np.abs(data.sites[site].used_error['ZYYR'] + 1j*data.sites[site].used_error['ZYYI'])
                    if jj == 4:
                        vals = data.sites[site].data['TZXR'] + 1j*data.sites[site].data['TZXI']
                        errors = np.abs(data.sites[site].used_error['TZXR'] + 1j*data.sites[site].used_error['TZXI'])
                    if jj == 5:
                        vals = data.sites[site].data['TZYR'] + 1j*data.sites[site].data['TZYI']
                        errors = np.abs(data.sites[site].used_error['TZYR'] + 1j*data.sites[site].used_error['TZYI'])
                    for kk in range(data.NP):
                        f.write('{:<10d} {:<10d} {:<10d} {:<14.6e} {:<14.6e} {:<14.6e}\n'.format(kk+1,
                                                                                                 ii+1,
                                                                                                 jj+1,
                                                                                                 np.real(vals[kk]),
                                                                                                 np.imag(vals[kk]),
                                                                                                 errors[kk]))

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

    def write_ModEM3D(data, out_file, use_elevation=False, include_flagged=True):
        if '.dat' not in out_file:
            out_file = ''.join([out_file, '.dat'])
        units = []
        data_type = []
        temp_inv_type = []
        actual_inv_type = copy.deepcopy(data.inv_type)
        with open(out_file, 'w') as f:
            z_title = '# Written using pyMT. UTM Zone: {}\n' + \
                      '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) ' + \
                      'Component Real Imag Error\n'.format(data.UTM_zone)
            pt_title = '# Written using pyMT. UTM Zone: {}\n' + \
                       '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) ' + \
                       'Component Value Error\n'.format(data.UTM_zone)
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
            elif data.inv_type == 6:
                data_type.append('> Phase_Tensor\n')
                units.append('> []\n')
                temp_inv_type.append(6)
            elif data.inv_type == 7:
                data_type.append('> Phase_Tensor\n')
                units.append('> []\n')
                temp_inv_type.append(6)
                data_type.append('> Full_Vertical_Components\n')
                units.append('> []\n')
                temp_inv_type.append(3)
            print(data.inv_type)
            print(temp_inv_type)
            # If inverting impedanaces or tipper
            if data.inv_type <= 5:
                for data_type_string, inv_type, unit in zip(data_type, temp_inv_type, units):
                    flagged_data = []
                    f.write(z_title)
                    f.write(data_type_string)
                    f.write('> exp(-i\\omega t)\n')
                    f.write(unit)
                    f.write('> {}\n'.format(data.azimuth))
                    f.write('> 0.0 0.0 0.0\n')
                    f.write('> {} {}\n'.format(data.NP, data.NS))
                    data.inv_type = inv_type
                    components_to_write = [component for component in data.used_components
                                           if 'i' not in component.lower()]
                    for ii, site_name in enumerate(data.site_names):
                        site = data.sites[site_name]
                        for jj, period in enumerate(data.periods):
                            for component in components_to_write:
                                component_code = component[:3]
                                if 'T' in component.upper():
                                    component_code = component_code[0] + component_code[2]
                                Z_real = site.data[component][jj]
                                Z_imag = site.data[component[:3] + 'I'][jj]
                                if abs(Z_real) > data.FLOAT_CAP:
                                    print('Exceedingly large value at station {}. Capping, but consider investigating...'.format(site_name))
                                    Z_real = np.sign(Z_real) * data.FLOAT_CAP
                                if abs(Z_imag) > data.FLOAT_CAP:
                                    print('Exceedingly large value at station {}. Capping, but consider investigating...'.format(site_name))
                                    Z_imag = np.sign(Z_imag) * data.FLOAT_CAP
                                if use_elevation:
                                    X, Y, Z = (data.locations[ii, 0],
                                               data.locations[ii, 1],
                                               site.locations.get('elev', 0))
                                else:
                                    X, Y, Z = (data.locations[ii, 0],
                                               data.locations[ii, 1],
                                               0)
                                # X, Y, Z = (data.locations[ii, 0],
                                #            data.locations[ii, 1],
                                #            site.locations.get('elev', 0))
                                # X, Y, Z = (data.locations[ii, 0],
                                #            data.locations[ii, 1],
                                #            site.locations.get('elev', 0))
                                Lat, Long = site.locations.get('Lat', 0), site.locations.get('Long', 0)
                                if True:
                                # if site.active_periods[jj]:
                                    line_out = ' '.join(['{:>14.7E} {:>14}',
                                                      '{:>8.3f} {:>8.3f}',
                                                      '{:>15.3f} {:>15.3f} {:>15.3f}',
                                                      '{:>6} {:>14.7E} {:>14.7E}',
                                                      '{:>14.7E}\n']).format(
                                                        period, site_name,
                                                        Lat, Long,
                                                        X, Y, Z,
                                                        component_code.upper(), Z_real, Z_imag,
                                                        min(max(site.used_error[component[:-1] + 'R'][jj],
                                                                site.used_error[component[:-1] + 'I'][jj]),
                                                            data.FLOAT_CAP))
                                    if site.used_error[component][jj] == data.REMOVE_FLAG:
                                        flagged_data.append(line_out)
                                    else:
                                        f.write(line_out)
                    if flagged_data and include_flagged:
                        f.write(''.join(flagged_data))
            # If inv_type is greater than 5, do phase tensors?
            else:
                for data_type_string, inv_type, unit in zip(data_type, temp_inv_type, units):
                    flagged_data = []
                    f.write(pt_title)
                    f.write(data_type_string)
                    f.write('> exp(-i\\omega t)\n')
                    f.write(unit)
                    f.write('> {}\n'.format(data.azimuth))
                    f.write('> 0.0 0.0 0.0\n')
                    f.write('> {} {}\n'.format(data.NP, data.NS))
                    # data.inv_type = inv_type
                    components_to_write = ['PTXX', 'PTXY', 'PTYX', 'PTYY']
                    for ii, site_name in enumerate(data.site_names):
                        site = data.sites[site_name]
                        for jj, period in enumerate(data.periods):
                            for component in components_to_write:
                                component_code = component[:]
                                # Remember X and Y are switched for Caldwell's def
                                if component in site.components:
                                    swapped_component = component.replace('X', '1')
                                    swapped_component = swapped_component.replace('Y', 'X')
                                    swapped_component = swapped_component.replace('1', 'Y')
                                    value = site.data[swapped_component][jj]
                                    error = site.used_error[swapped_component][jj]
                                else:
                                    # Remember X and Y are switched for Caldwell's def
                                    if component == 'PTXX':  # PTXX
                                        value = site.phase_tensors[jj].phi[1, 1]
                                        error = site.phase_tensors[jj].phi_error[1, 1]
                                    elif component == 'PTXY':  # PTXY
                                        value = site.phase_tensors[jj].phi[1, 0]
                                        error = site.phase_tensors[jj].phi_error[1, 0]
                                    elif component == 'PTYX':  # PTYX
                                        value = site.phase_tensors[jj].phi[0, 1]
                                        error = site.phase_tensors[jj].phi_error[0, 1]
                                    elif component == 'PTYY':  # PTYY
                                        value = site.phase_tensors[jj].phi[0, 0]
                                        error = site.phase_tensors[jj].phi_error[0, 0]
                                X, Y, Z = (site.locations['X'],
                                           site.locations['Y'],
                                           site.locations.get('elev', 0))
                                X, Y, Z = (data.locations[ii, 0],
                                           data.locations[ii, 1],
                                           site.locations.get('elev', 0))
                                Lat, Long = site.locations.get('Lat', 0), site.locations.get('Long', 0)
                                line_out = ' '.join(['{:>14.7E} {:>14}',
                                                  '{:>8.3f} {:>8.3f}',
                                                  '{:>15.3f} {:>15.3f} {:>15.3f}',
                                                  '{:>6} {:>14.7E} {:>14.7E}\n']).format(
                                                    period, site_name,
                                                    Lat, Long,
                                                    X, Y, Z,
                                                    component_code.upper(), value,
                                                    error)
                                if error == data.REMOVE_FLAG:
                                    flagged_data.append(line_out)
                                else:
                                    f.write(line_out)
                if flagged_data and include_flagged:
                    f.write(''.join(flagged_data))
        data.inv_type = actual_inv_type

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
            f.write('UTM of x, y origin (UTM zone, N, E, 2D strike): {:>10} {:>10.4f} {:>10.4f} {:>10.2f}\n'.format(
                    data.UTM_zone, data.origin[0], data.origin[1], strike))
            f.write('# MT Frequencies: {}\n'.format(data.NP))
            for freq in frequencies:
                f.write('  {}\n'.format(freq))
            f.write('# MT Receivers: {}\n'.format(data.NS))
            f.write('!{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}{:>15}\n'.format(
                    'X', 'Y', 'Z', 'Theta', 'Alpha', 'Beta', 'Length', 'SolveStatic', 'Name'))
            for site in data.site_names:
                f.write(' {:>15.4f}{:>15.4f}{:>15.4f}{:>15.4f}{:>15.4f}{:>15.4f}{:>15.4f}{:>15.4f}{:>15}\n'.format(
                    utils.truncate(data.sites[site].locations['Y']),
                    utils.truncate(data.sites[site].locations['X']),
                    getattr(data.sites[site].locations, 'elev', 0),
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
                        if not component.lower()[:3] == 'tzx':
                            data_point = data.sites[site].data[component][jj]
                            if component.endswith('I'):
                                data_point = -1 * data_point
                            f.write('{:>15} {:>15} {:>15} {:>15} {:>15.6g} {:>15.6g}\n'.format(
                                    data_type_lookup[component],
                                    jj + 1,
                                    ii + 1,
                                    ii + 1,
                                    data_point,
                                    # data.sites[site].data[component][jj],
                                    data.sites[site].used_error[component][jj]))

    def write_occam(data, outfile):
        # Currently only writes Rho and Pha (can't choose) and sets static errors
        with open(outfile, 'w') as f:
            f.write('  FORMAT:         OCCAM2MTDATA_1.0\n')
            f.write('  MODEL:          {}\n'.format(outfile))
            f.write('  SITES:{:>17g}\n'.format(len(data.site_names)))
            data_types = {'ZXYR': 13, 'ZXYI': 14,
                          'ZYXR': 15, 'ZYXI': 16,
                          'TZYR': 3, 'TZYI': 4,
                          'RhoXY': 1, 'RhoYX': 5,
                          'PhaXY': 2, 'PhaYX': 6}
            exclude_components = set(('ZXXR', 'ZXXI',
                                      'ZYYR', 'ZYYI',
                                      'TZXR', 'TZXI'))
            NS, NP, NR = (len(data.site_names),
                          len(data.periods),
                          len(set(data.used_components) - exclude_components))
            num_data = NS * NP * 4
            X = data.locations[:, 0] - np.min(data.locations[:, 0])
            frequencies = utils.truncate(1 / data.periods)
            for ii, site in enumerate(data.site_names):
                f.write('{}\n'.format(site))
            f.write('  OFFSETS (M):\n')
            for x in X:
                f.write('{:>15.8E}'.format(x))
            f.write('\n')
            f.write('  FREQUENCIES:{:>13g}\n'.format(len(frequencies)))
            for freq in frequencies:
                f.write('{:>15.8E}'.format(freq))
            f.write('\n')
            f.write('  DATA BLOCKS:{:>13g}\n'.format(num_data))
            f.write('SITE   FREQ   DATA TYPE     DATUM       ERR\n')
            for ii, site in enumerate(data.site_names):
                for comp in ('RhoXY', 'PhaXY', 'RhoYX', 'PhaYX'):
                    if comp not in exclude_components:
                        comp_code = data_types[comp]
                        if comp[:3].lower() == 'rho':
                            calc_data, calc_err, calclog10_err = utils.compute_rho(data.sites[site],
                                                                                   calc_comp=comp,
                                                                                   errtype='used_error')
                            calc_data = np.log10(calc_data)
                            calc_error = 0.08
                        elif comp[:3].lower() == 'pha':
                            calc_data, calclog10_err = utils.compute_phase(data.sites[site],
                                                                           calc_comp=comp,
                                                                           errtype='used_error',
                                                                           wrap=1)
                            calc_error = 5
                        for jj, freq in enumerate(frequencies):
                            data_point = calc_data[jj]
                            error_point = calc_error
                            f.write('{:>4g}{:>6g}{:>12g}{:>15.5f}{:>15.5f}\n'.format(ii + 1,
                                                                                     jj + 1,
                                                                                     comp_code,
                                                                                     data_point,
                                                                                     error_point))

    def write_ModEM2D(data, out_file):
        if '.dat' not in out_file:
            out_file = ''.join([out_file, '.dat'])
        units = []
        data_type = []
        temp_inv_type = []
        actual_inv_type = copy.deepcopy(data.inv_type)
        with open(out_file, 'w') as f:
            z_title = '# Written using pyMT. UTM Zone: {}\n' + \
                      '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) ' + \
                      'Component Real Imag Error\n'.format(data.UTM_zone)
            if data.inv_type == 10:
                data_type.append('> TE_Impedance\n')
                units.append('> Ohm\n')
                data_type.append('> TM_Impedance\n')
                units.append('> Ohm\n')
                temp_inv_type.append(8)
                temp_inv_type.append(9)
            elif data.inv_type == 8:
                data_type.append('> TE_Impedance\n')
                units.append('> Ohm\n')
                temp_inv_type.append(8)
            elif data.inv_type == 9:
                data_type.append('> TM_Impedance\n')
                units.append('> Ohm\n')
                temp_inv_type.append(9)
            # If inverting impedances
            # If locations haven't been projected
            azi = copy.deepcopy(data.azimuth)
            if np.any(data.locations[:, 0]):
                # Reset the data locations
                data.rotate_sites(azi=0)
                # Rotate the data to coincide with the chosen direction
                for site_name in data.site_names:
                    data.sites[site_name].rotate_data(azi)  # There is a negative in the data_structures code that has to be countered here
                # Project the station locations onto the line
                xy = np.fliplr(data.locations)
                xy = utils.project_to_line(xy, azi=(azi + 90) % 180)
                xy, center = utils.center_locs(xy)
                center = xy[0, :]
                for ii, (x, y) in enumerate(xy):
                    if x < center[0]:
                        if y < center[1]:
                            center = xy[ii, :]
                y_locs = []
                for x, y in xy:
                    y_locs.append(np.sqrt((x - center[0]) ** 2 + (y - center[0]) ** 2))
            else:
                y_locs = data.locations[:, 1]
            for data_type_string, inv_type, unit in zip(data_type, temp_inv_type, units):
                f.write(z_title)
                f.write(data_type_string)
                f.write('> exp(-i\\omega t)\n')
                f.write(unit)
                f.write('> {}\n'.format(azi))
                f.write('> 0.0 0.0 0.0\n')
                f.write('> {} {}\n'.format(data.NP, data.NS))
                data.inv_type = inv_type
                components_to_write = [component for component in data.used_components
                                       if 'i' not in component.lower()]
                for ii, site_name in enumerate(data.site_names):
                    site = data.sites[site_name]
                    for jj, period in enumerate(data.periods):
                        for component in components_to_write:
                            if component[:3] == 'ZXY':
                                component_code = 'TE'
                            elif component[:3] == 'ZYX':
                                component_code = 'TM'
                            Z_real = site.data[component][jj]
                            Z_imag = site.data[component[:3] + 'I'][jj]
                            # X, Y, Z = (site.locations['X'],
                            #            site.locations['Y'],
                            #            site.locations.get('elev', 0))
                            # X, Y, Z = (data.locations[ii, 0],
                            #            data.locations[ii, 1],
                            #            site.locations.get('elev', 0))
                            Lat, Long = site.locations.get('Lat', 0), site.locations.get('Long', 0)
                            f.write(' '.join(['{:>14.7E} {:>14}',
                                              '{:>8.3f} {:>8.3f}',
                                              '{:>15.3f} {:>15.3f} {:>15.3f}',
                                              '{:>6} {:>14.7E} {:>14.7E}',
                                              '{:>14.7E}\n']).format(
                                    period, site_name,
                                    Lat, Long,
                                    0, y_locs[ii], 0,
                                    component_code.upper(), Z_real, Z_imag,
                                    max(site.used_error[component[:-1] + 'R'][jj],
                                        site.used_error[component[:-1] + 'I'][jj])))

    def write_gofem(data, out_file):
        comp_dict = {'ZXXR': 'RealZxx',
                     'ZXXI': 'ImagZxx',
                     'ZXYR': 'RealZxy',
                     'ZXYI': 'ImagZxy',
                     'ZYXR': 'RealZyx',
                     'ZYXI': 'ImagZyx',
                     'ZYYR': 'RealZyy',
                     'ZYYI': 'ImagZyy',
                     'TZXR': 'RealTzx',
                     'TZXI': 'ImagTzx',
                     'TZYR': 'RealTzy',
                     'TZYI': 'ImagTzy'}
        print('Flagged data removed from gofem data file')
        msg_written = 0
        with open(out_file, 'w') as f:
            f.write('# DataType Frequency Source Receiver Value Error\n')
            for ii, period in enumerate(data.periods):
                for site_name in data.site_names:
                    for comp in data.used_components:
                        freq = 1 / period
                        data_point = data.sites[site_name].data[comp][ii]
                        error_point = abs(data.sites[site_name].used_error[comp][ii])
                        # GoFEM doesn't like this with the tippers - should they not be multiplied by -1?
                        # if comp.lower().startswith('z') and comp.lower().endswith('i'):
                        if comp.lower().endswith('i'):
                            data_point = -1 * data_point
                        
                        if error_point != data.REMOVE_FLAG:
                            f.write('{} {:>12.6e} Plane_wave {} {:>12.6e} {:>12.6e}\n'.format(comp_dict[comp],
                                                                                              freq,
                                                                                              site_name,
                                                                                              data_point,
                                                                                              error_point))
                        else:
                            if not msg_written:
                                print('Flagged data removed from gofem data file')
                                msg_written = 1


    if file_format.lower() == 'wsinv3dmt':
        write_ws(data, outfile, to_write)
    elif file_format.lower() == 'modem':
        if data.dimensionality.lower() == '3d':
            write_ModEM3D(data, outfile, use_elevation, include_flagged)
        elif data.dimensionality.lower() == '2d' or data.inv_type in (8, 9, 10):
            write_ModEM2D(data, outfile)
        else:
            print('Unable to write data; dimensionality not defined')
    elif file_format.lower() == 'mare2dem':
        write_MARE2DEM(data, outfile)
    elif file_format.lower() == 'occam':
        write_occam(data, outfile)
    elif file_format.lower() == 'em3dani':
        write_em3dani(data, outfile)
    elif file_format.lower() == 'gofem':
        write_gofem(data, outfile)
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


def write_list(data, outfile):
    if '.lst' not in outfile:
        outfile = ''.join([outfile, '.lst'])
    with open(outfile, 'w') as f:
        f.write('{}\n'.format(len(data.site_names)))
        for site in data.site_names[:-1]:
            f.write('{}\n'.format(''.join([site, '.dat'])))
        f.write('{}'.format(''.join([data.site_names[-1], '.dat'])))


def write_model(model, outfile, file_format='modem', use_anisotropy=False, use_log=True, use_resistivity=True, n_param=3):
    def to_em3dani(model, outfile, use_log, use_resistivity, use_anisotropy):
        # debug_print([use_log, use_resistivity], 'E:/phd/NextCloud/data/synthetics/EM3DANI/wst/aniso8/debug.log')
        value_dict = {'sigma': 'vals',
                      'sigmax': 'rho_x', 'sigmay': 'rho_y', 'sigmaz': 'rho_z',
                      'slant': 'slant', 'dip': 'dip', 'strike': 'strike'}
        if not outfile.lower().endswith('.mod'):
            outfile += '.mod'
        with open(outfile, 'w') as f:
            f.write('# Format:      EM3DModelFile_1.0\n')
            f.write('# Description:     Written from pyMT\n')
            f.write('NX:     {}\n'.format(model.nx))
            for xx in model.xCS:
                f.write('{:<14.0f} '.format(xx))
            f.write('\n')
            f.write('NY:     {}\n'.format(model.ny))
            for yy in model.yCS:
                f.write('{:<14.0f} '.format(yy))
            f.write('\n')
            f.write('NAIR:     7\n')
            for iz in [100, 300, 1000, 3000, 10000, 30000, 100000]:
                f.write('{:<14.0f} '.format(iz))
            f.write('\n')
            f.write('NZ:     {}\n'.format(model.nz))
            for iz in model.zCS:
                f.write('{:<14.0f} '.format(iz))
            f.write('\n')
            if use_resistivity:
                f.write('Resistivity Type:  Resistivity\n')
            else:
                f.write('Resistivity Type:  Conductivity\n')
            if use_log:
                f.write('Model Type:        Log\n')
            else:
                f.write('Model Type:        Linear\n')
            if len(model.rho_y) > 1:
                val_names = ['sigmax', 'sigmay', 'sigmaz']
                use_anisotropy = True
            else:
                val_names = ['sigma']
            if len(model.strike) > 1:
                val_names += ['strike']
                val_names += ['dip']
                val_names += ['slant']
            if use_anisotropy:
                f.write('Anisotropy Type: Anisotropy\n')
            for name in val_names:
                f.write('{}:\n'.format(name))
                values = getattr(model, value_dict[name])
                # for val in values:
                for iz in range(model.nz):
                    for iy in range(model.ny):
                        for ix in range(model.nx):
                            if 'sigma' in name:
                                if use_resistivity:
                                    val = values[ix, iy, iz]
                                else:
                                    val = 1 / values[ix, iy, iz]
                                if use_log and 'sigma' in name:
                                    val = np.log10(val)
                                # f.write('{:<10.5f} '.format(np.log10(values[ix, iy, iz])))
                                f.write('{:<10.5f} '.format(val))
                            else:
                                f.write('{:<10.5f} '.format((values[ix, iy, iz])))
                    f.write('\n')
            ox, oy = np.sum(model.xCS) / 2, np.sum(model.yCS) / 2
            f.write('Origin: {:<14.2f}  {:<14.2f}  0.0'.format(ox, oy))

    def to_ubc(model, outfile):
        file_name, ext = os.path.splitext(outfile)
        with open(file_name + '.msh', 'w') as f:
            f.write('{:<5d}{:<5d}{:<5d}\n'.format(model.ny, model.nx, model.nz))
            f.write('{:<14.7f} {:<14.7f} {:<14.7f}\n'.format(model.dy[0],
                                                             model.dx[0],
                                                             model.dz[0]))
            for y in model.yCS:
                f.write('{:<10.7f} '.format(y))
            f.write('\n')
            for x in model.xCS:
                f.write('{:<10.7f} '.format(x))
            f.write('\n')
            for z in model.zCS:
                f.write('{:<10.7f} '.format(z))

        with open(file_name + '.res', 'w') as f:
            for ix in range(model.nx):
                for iy in range(model.ny):
                    for iz in range(model.nz):
                        f.write('{:<10.7f}\n'.format(model.vals[ix, iy, iz]))

    def to_mt3dani(model, outfile, use_log, use_resistivity, use_anisotropy, n_param):
        if '.zani' not in outfile:
            outfile = ''.join([outfile, '.zani'])
        is_half_space = int(model.is_half_space())
        if use_log:
            header_four = 'LOGE'
        else:
            header_four = 'LINEAR'
        with open(outfile, 'w') as f:
            f.write('{}\n'.format('# ' + outfile))
            f.write('{} {} {} {} {}\n'.format(model.nx, model.ny, model.nz, 0, header_four))
            for x in model.xCS:
                f.write('{:<10.7f}  '.format(x))
            f.write('\n')
            for y in model.yCS:
                f.write('{:<10.7f}  '.format(y))
            f.write('\n')
            for z in model.zCS:
                f.write('{:<10.7f}  '.format(z))
            f.write('\n')

            param = ['rho_x', 'rho_y', 'rho_z', 'strike', 'dip', 'slant']
            for pp in range(n_param):
                if header_four == 'LOGE':
                    if np.any(model.vals < 0):
                        print('Negative values detected in model.')
                        print('I hope you know what you\'re doing.')
                        vals = np.sign(getattr(model, param[pp], model.vals)) * np.log(np.abs(getattr(model, param[pp], model.vals)))
                        vals = np.nan_to_num(vals)
                    else:
                        vals = getattr(model, param[pp], model.vals)
                        if len(vals) == 0:
                            vals = np.zeros(shape=model.vals.shape)
                        if pp < 3:
                            vals = np.log(vals)
                else:
                    vals = get(model, param[pp], model.vals)
                for zz in range(model.nz):
                    for yy in range(model.ny):
                        for xx in range(model.nx):
                            # f.write('{:<10.7E}\n'.format(model.vals[model.nx - xx - 1, yy, zz]))
                            f.write('{:<14.5E}'.format(vals[model.nx - xx - 1, yy, zz]))
                        f.write('\n')
                    f.write('\n')
                f.write('\n')

    def to_modem(model, outfile, file_format):
        if '.model' not in outfile:
            outfile = ''.join([outfile, '.model'])
        is_half_space = int(model.is_half_space())
        if file_format.lower() == 'modem':
            header_four = 'LOGE'
            is_half_space = 0
        else:
            header_four = ''
        with open(outfile, 'w') as f:
            f.write('{}\n'.format('# ' + outfile))
            f.write('{} {} {} {} {}\n'.format(model.nx, model.ny, model.nz, is_half_space, header_four))
            for x in model.xCS:
                f.write('{:<10.3f}  '.format(x))
            f.write('\n')
            for y in model.yCS:
                f.write('{:<10.3f}  '.format(y))
            f.write('\n')
            for z in model.zCS:
                f.write('{:<10.3f}  '.format(z))
            f.write('\n\n')
            if header_four == 'LOGE':
                if np.any(model.vals < 0):
                    print('Negative values detected in model.')
                    print('I hope you know what you\'re doing.')
                    vals = np.sign(model.vals) * np.log(np.abs(model.vals))
                    vals = np.nan_to_num(vals)
                else:
                    vals = np.log(model.vals)
            else:
                vals = model.vals
            if is_half_space and header_four != 'LOGE':
                f.write(str(model.vals[0, 0, 0]))
            else:
                for zz in range(model.nz):
                    for yy in range(model.ny):
                        for xx in range(model.nx):
                            # f.write('{:<10.7E}\n'.format(model.vals[model.nx - xx - 1, yy, zz]))
                            f.write('{:<13.5E}'.format(vals[model.nx - xx - 1, yy, zz]))
                        f.write('\n')
                    f.write('\n')
            # Write bottom left corner coordinate and rotation angle (dummies right now)
            # f.write('\n')
            f.write('{:<15.3f}{:<15.3f}{:<15.3f}\n'.format(model.dx[0], model.dy[0], model.dz[0]))
            f.write('{:<6.3f}'.format(0))
    def write_modem_2d(model, outfile):
        if '.model' not in outfile:
            outfile = ''.join([outfile, '.model'])
        # is_half_space = int(model.is_half_space())
        # if file_format.lower() == 'modem':
        header_four = 'LOGE'
        is_half_space = 0
        # else:
            # header_four = ''
        with open(outfile, 'w') as f:
            f.write('{} {} {}\n'.format(model.ny, model.nz, header_four))
            # for x in model.xCS:
                # f.write('{:<10.7f}  '.format(x))
            f.write('\n')
            for y in model.yCS:
                f.write('{:<10.7f}  '.format(y))
            f.write('\n')
            for z in model.zCS:
                f.write('{:<10.7f}  '.format(z))
            f.write('\n')
            if header_four == 'LOGE':
                if np.any(model.vals < 0):
                    print('Negative values detected in model.')
                    print('I hope you know what you\'re doing.')
                    vals = np.sign(model.vals) * np.log(np.abs(model.vals))
                    vals = np.nan_to_num(vals)
                else:
                    vals = np.log(model.vals)
            else:
                vals = model.vals
            if is_half_space and header_four != 'LOGE':
                f.write(str(model.vals[0, 0, 0]))
            else:
                for zz in range(model.nz):
                    for yy in range(model.ny):
                    # for xx in range(model.nx):
                        # f.write('{:<10.7E}\n'.format(np.log(model.vals[model.nx - xx - 1, 0, zz])))
                        f.write('{:<10.7f}\n'.format(vals[0, yy, zz]))

    def to_csv(model, outfile):
        if not (outfile.endswith('csv')):
            outfile += '.csv'
        x, y, z = (utils.edge2center(arr) for arr in (model.dx, model.dy, model.dz))
        with open(outfile, 'w') as f:
            f.write('Easting, Northing, Depth, Resistivity\n')
            for ix in range(model.nx):
                for iy in range(model.ny):
                    for iz in range(model.nz):
                        f.write('{:>13.4f},{:>13.4f},{:>13.4f},{:>13.4f}\n'.format(y[iy],
                                                                                   x[ix],
                                                                                   -z[iz],
                                                                                   model.vals[ix, iy, iz]))

    if file_format.lower() in ('modem', 'wsinv', 'wsinv3dmt'):
        to_modem(model=model, outfile=outfile, file_format=file_format)
    elif file_format.lower() in ('modem2d'):
        write_modem_2d(model=model, outfile=outfile)
    elif file_format.lower() in ('ubc', 'ubc-gif'):
        to_ubc(model=model, outfile=outfile)
    elif file_format.lower() in ('csv'):
        to_csv(model=model, outfile=outfile)
    elif file_format.lower() in ('em3dani'):
        to_em3dani(model=model,
                   outfile=outfile,
                   use_log=use_log, 
                   use_resistivity=use_resistivity,
                   use_anisotropy=use_anisotropy)
    elif file_format.lower() in ('mt3dani'):
        to_mt3dani(model=model,
                   outfile=outfile,
                   use_log=use_log,
                   use_resistivity=use_resistivity,
                   use_anisotropy=use_anisotropy,
                   n_param=n_param)
    else:
        print('File format {} not supported'.format(file_format))
        print('Supported formats are: ')
        print('ModEM, WSINV3DMT, UBC-GIF, CSV, EM3DANI, MT3DANI')
        return

def write_phase_tensors(data, out_file, verbose=False, scale_factor=1/50, period_idx=None):

    if not out_file.endswith('.csv'):
        out_file += '.csv'
    print('Writing phase tensor data to {}'.format(out_file))
    with open(out_file, 'w') as f:
        header = ['Site', 'Period', 'Latitude', 'Longitude', 'Azimuth',
                  'Phi_min', 'Phi_max', 'Phi_min_scaled', 'Phi_max_scaled', 'Phi_split']
        if verbose:
            header += ['Phi_1', 'Phi_2', 'Phi_3', 'Det_Phi', 'Alpha', 'Beta', 'Lambda']
        f.write(','.join(header))
        f.write('\n')
        try:
            test = data.sites[data.site_names[0]].locations['Lat'], data.sites[data.site_names[0]].locations['Long']
            X_key, Y_key = 'Lat', 'Long'
        except KeyError:
            X_key, Y_key = 'Y', 'X'
        X_all = [site.locations['X'] for site in data.sites.values()]
        Y_all = [site.locations['Y'] for site in data.sites.values()]
        scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
                        (np.max(Y_all) - np.min(Y_all)) ** 2)
        scale *= scale_factor
        for site_name in data.site_names:
            site = data.sites[site_name]
            X, Y = site.locations[X_key], site.locations[Y_key]
            if period_idx is None:
                # period_idx = list(range(site.NP))
                period_idx = site.periods
            for ii, period in enumerate(site.periods):
                if min(abs(period - period_idx)) < period * 0.1: # in period_idx:
                    phi_max = 1 * scale
                    phi_min = scale * site.phase_tensors[ii].phi_min / site.phase_tensors[ii].phi_max
                    f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(site_name,
                                                            period,
                                                            X,
                                                            Y,
                                                            -np.rad2deg((site.phase_tensors[ii].azimuth)) + 90,
                                                            site.phase_tensors[ii].phi_min,
                                                            site.phase_tensors[ii].phi_max,
                                                            phi_min,
                                                            phi_max,
                                                            site.phase_tensors[ii].phi_max - site.phase_tensors[ii].phi_min))
                    if verbose:
                        f.write(', {}, {}, {}, {}, {}, {}, {}\n'.format(np.rad2deg(np.arctan(site.phase_tensors[ii].phi_1)),
                                                                        np.rad2deg(np.arctan(site.phase_tensors[ii].phi_2)),
                                                                        np.rad2deg(np.arctan(site.phase_tensors[ii].phi_3)),
                                                                        np.rad2deg(np.arctan(site.phase_tensors[ii].det_phi)),
                                                                        np.rad2deg((site.phase_tensors[ii].alpha)),
                                                                        np.rad2deg((site.phase_tensors[ii].beta)),
                                                                        site.phase_tensors[ii].Lambda))
                    else:
                        f.write('\n')


def write_induction_arrows(data, out_file, period_idx=None, verbose=False, scale_factor=1):
    if not out_file.endswith('.csv'):
        out_file += '.csv'
    with open(out_file, 'w') as f:
        header = header = ['Site', 'Period', 'Latitude', 'Longitude',
                           'TXR', 'TYR', 'TXI', 'TYI', 'Rev. TXR', 'Rev. TYR', 'Rev. TXI', 'Rev. TYI',
                           'Azimuth_R', 'Azimuth_I', 'Magnitude_R', 'Magnitude_I']

        try:
            test = data.sites[data.site_names[0]].locations['Lat'], data.sites[data.site_names[0]].locations['Long']
            X_key, Y_key = 'Lat', 'Long'
        except KeyError:
            X_key, Y_key = 'Y', 'X'
        f.write(','.join(header))
        f.write('\n')
        for site_name in data.site_names:
            site = data.sites[site_name]
            X, Y = site.locations[X_key], site.locations[Y_key]
            if period_idx is None:
                period_idx = site.periods
            if set(site.TIPPER_COMPONENTS).issubset(set(site.components)):
                for ii, period in enumerate(site.periods):
                    if min(abs(period - period_idx)) < period * 0.1:
                        azi_R = np.rad2deg(np.arctan2(-site.data['TZYR'][ii], -site.data['TZXR'][ii]))
                        azi_I = np.rad2deg(np.arctan2(-site.data['TZYI'][ii], -site.data['TZXI'][ii]))
                        mag_R = np.sqrt(site.data['TZXR'][ii] ** 2 + site.data['TZYR'][ii] ** 2)
                        mag_I = np.sqrt(site.data['TZXI'][ii] ** 2 + site.data['TZYI'][ii] ** 2)
                        f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(site_name,
                                                                                          period,
                                                                                          X,
                                                                                          Y,
                                                                                          site.data['TZXR'][ii],
                                                                                          site.data['TZYR'][ii],
                                                                                          site.data['TZXI'][ii],
                                                                                          site.data['TZYI'][ii],
                                                                                          -site.data['TZXR'][ii],
                                                                                          -site.data['TZYR'][ii],
                                                                                          -site.data['TZXI'][ii],
                                                                                          -site.data['TZYI'][ii],
                                                                                          azi_R,
                                                                                          azi_I,
                                                                                          mag_R,
                                                                                          mag_I))
            else:
                print('No tipper in {}'.format(site_name))






