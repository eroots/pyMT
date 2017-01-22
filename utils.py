"""Summary
"""
import numpy as np
import os
from math import log10, floor


def percdiff(val1, val2):
    """Summary
    
    Args:
        val1 (TYPE): Description
        val2 (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return abs(val1 - val2) / (0.5 * (val1 + val2))


def generate_mesh(site_locs, min_x, model=None,
                  num_pads=None, pad_mult=None, DEBUG=True):
    """Summary
    
    Args:
        site_locs (TYPE): Description
        min_x (TYPE): Description
        model (None, optional): Description
        num_pads (None, optional): Description
        pad_mult (None, optional): Description
        DEBUG (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    MAX_X = 150
    xlocs = site_locs  # South-North
    # ylocs = site_locs[:, 1]  # West-East
    xloc_sort = np.sort(xlocs)
    # yloc_sort = np.sort(ylocs)
    xoff = xloc_sort[0] + (xloc_sort[-1] - xloc_sort[0]) / 2
    xloc_sort = (xloc_sort - xoff).astype(int)
    x_seperation = np.diff(xloc_sort)
    max_sep = np.max(x_seperation)
    xmesh = np.zeros(MAX_X)
    # median_seperation = np.median(x_seperation)
    avg_sep = np.mean(x_seperation)
    max_xmin = 3 * avg_sep
    if min_x > max_xmin:
        print('Minimum cell size shouldn\'t be more than {}'.format(max_xmin))
        return
    dist_without_mesh = 0
    is_right_ofmesh = 0
    imesh = 1
    xmesh[0] = xloc_sort[0] - min_x * 0.5
    for ii in range(len(xloc_sort) - 1):
        # Look to station location on right
        dist_right = xloc_sort[ii + 1] - xloc_sort[ii]
        ifact = 0
        dist = 0
        while dist < dist_right:
            ifact += 1
            dist += min_x + min_x * ifact
        ifact -= ifact
        if ifact >= 2:
            if (imesh + ifact * 2 + 1 > MAX_X):
                resp = input('Number of cells in X direction exceeds {}. Continue? (y/n)'.format(MAX_X))
                if resp == 'n':
                    return
            for jj in range(1, max(ifact - 2, 1) + 1):
                imesh += 1
                xmesh[imesh - 1] = xmesh[imesh - 2] + min_x * jj
            imesh += 1
            xmesh[imesh - 1] = xloc_sort[ii] + (xloc_sort[ii + 1] - xloc_sort[ii]) / 2
            imesh += ifact - 1
            imesh_save = imesh
            xmesh[imesh - 1] = xloc_sort[ii + 1] - min_x / 2
            for jj in range(1, ifact - 1):
                imesh -= 1
                xmesh[imesh - 1] = xmesh[imesh] - min_x * jj
            imesh = imesh_save
            dist_without_mesh = 0
            is_right_ofmesh = ii + 1
        elif (dist_right >= min_x) and (dist_right <= 5 * min_x):
            if DEBUG:
                print('Gap is small, splitting sites')
            if (imesh + 1 >= MAX_X):
                resp = input('Number of cells in X direction exceeds {}. Continue? (y/n)'.format(MAX_X))
                if resp == 'n':
                    return
            imesh += 1
            xmesh[imesh - 1] = xloc_sort[ii] + (xloc_sort[ii + 1] - xloc_sort[ii]) / 2
            dist_without_mesh = 0
            is_right_ofmesh = ii + 1
        else:
            # no room for mesh?
            if DEBUG:
                print('Sites too close, splitting mesh')
            dist_without_mesh = xloc_sort[ii + 1] - xmesh[imesh - 1]
        if dist_without_mesh > 2 * min_x:
            if DEBUG:
                print("Gone too far without adding mesh")
            max_sep = 0
            # if is_right_ofmesh == 0:
            # is_right_ofmesh = 1
            for is_check in range(is_right_ofmesh, ii + 1):
                check_sep = xloc_sort[is_check + 1] - xloc_sort[is_check]
                if check_sep > max_sep:
                    is_save = is_check
                    max_sep = check_sep
            if imesh + 1 > MAX_X:
                resp = input('Number of cells in X direction exceeds {}. Continue? (y/n)'.format(MAX_X))
                if resp == 'n':
                    return
            imesh += 1
            xmesh[imesh - 1] = xloc_sort[is_save] + (xloc_sort[is_save + 1] - xloc_sort[is_save]) / 2
            dist_without_mesh = 0
            is_right_ofmesh = is_save + 1
    imesh += 1
    if imesh > MAX_X:
        resp = input('Number of cells in X direction exceeds {}. Continue? (y/n)'.format(MAX_X))
        if resp == 'n':
            return
    xmesh[imesh - 1] = xloc_sort[-1] + xloc_sort[-1] - xmesh[imesh - 2]
    nmesh = imesh
    return xmesh[:nmesh], nmesh


def enforce_input(**types):
    """Summary
    
    Args:
        **types (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    def decorator(func):
        """Summary
        
        Args:
            func (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        def new_func(*args, **kwargs):
            """Summary
            
            Args:
                *args (TYPE): Description
                **kwargs (TYPE): Description
            
            Returns:
                TYPE: Description
            """
            newargs = {}
            for k, t in types.items():
                arg = kwargs.get(k, None)
                if isinstance(arg, t) or arg is None:
                    newargs.update({k: arg})
                else:
                    if t == list:
                        newargs.update({k: to_list(kwargs[k])})
                    elif t is np.ndarray:
                        newargs.update({k: np.array(kwargs[k])})
                    else:
                        newargs.update({k: t(kwargs[k])})
            return func(*args, **newargs)
        return new_func
    return decorator


def enforce_output(*types):
    """Summary
    
    Args:
        *types (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    def decorator(func):
        """Summary
        
        Args:
            func (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        def new_outputs(*args, **kwargs):
            """Summary
            
            Args:
                *args (TYPE): Description
                **kwargs (TYPE): Description
            
            Returns:
                TYPE: Description
            """
            newouts = []
            outs = func(*args, **kwargs)
            for (t, a) in zip(types, outs):
                if isinstance(a, t):
                    newouts.append(a)
                else:
                    if t == list:
                        newouts.append(to_list(a))
                    elif t == np.array:
                        newouts.append(np.array(a))
                    else:
                        newouts.append(t(a))
            return newouts
        return new_outputs
    return decorator


def closest_periods(available_p, wanted_p):
    """Summary
    
    Args:
        available_p (TYPE): Description
        wanted_p (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    retval = []
    for p in wanted_p:
        ind = np.argmin([abs(ap - p) for ap in available_p])
        retval.append(available_p[ind])
    retval = sorted(set(retval))
    if isinstance(available_p, np.ndarray):
        retval = np.array(retval)
    return retval


def list_or_numpy(func):
    """Summary
    
    Args:
        func (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    def inner(obj):
        """Summary
        
        Args:
            obj (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        made_change = False
        if isinstance(obj, np.ndarray):
            obj = [x for x in obj]
            made_change = True
        retval = func(obj)
        if made_change:
            retval = np.array(retval)
        return retval
    return inner


def check_file(outfile):
    """Summary
    
    Args:
        outfile (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    try:
        open(outfile, 'r')
    except FileNotFoundError:
        return False
    else:
        return True


def row_or_col(func):
    """Summary
    
    Args:
        func (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    def inner(vec):
        """Summary
        
        Args:
            vec (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        made_change = False
        if np.shape(vec)[0] < np.shape(vec)[1]:
            vec = np.transpose(vec)
            made_change = True
        retval = func(vec)
        if made_change:
            retval = np.transpose(retval)
        return retval
    return inner


def geo2utm(Lat, Long):
    """Summary
    
    Args:
        Lat (TYPE): Description
        Long (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    # AA = 6378206.4
    # ECC2 = 0.00676866
    AA = 6378137
    ECC2 = 0.00669438
    K0 = 0.9996
    origin = 999
    if Long == 0:
        origin = -3
    else:
        origin = float(int(Long / 6) * 6) + 3 * int(Long / 6) / abs(int(Long / 6))
    LatR, LongR = np.deg2rad([Lat, Long])
    eccPrimeSquared = ECC2 / (1 - ECC2)
    N = AA / (np.sqrt(1 - ECC2 * np.sin(Lat) ** 2.0))
    T = np.tan(LatR) ** 2
    C = eccPrimeSquared * np.cos(LatR) ** 2
    A = np.cos(LatR) * np.deg2rad((Long - origin))
    M = AA * ((1 -
               ECC2 / 4 -
               3 * ECC2 * ECC2 / 64 -
               5 * ECC2 * ECC2 * ECC2 / 256) * LatR -
              (3 * ECC2 / 8 +
               3 * ECC2 * ECC2 / 32 +
               45 * ECC2 * ECC2 * ECC2 / 1024) * np.sin(2 * LatR) +
              (15 * ECC2 * ECC2 / 256 + 45 * ECC2 * ECC2 * ECC2 / 1024) * np.sin(4 * LatR) -
              (35 * ECC2 * ECC2 * ECC2 / 3072) * np.sin(6 * LatR))
    # M = 111132.0894 * Lat - 16216.94 * np.sin(2 * LatR) + 17.21 * \
    #     np.sin(4 * LatR) - 0.02 * np.sin(6 * LatR)
    Easting = K0 * N * \
        (A + (1 - T + C) * ((A ** 3) / 6) +
         (5 - 18 * T ** 3 + 72 * C - 58 * eccPrimeSquared) *
            (A ** 5) / 120) + 500000

    Northing = K0 * \
        (M + N * np.tan(LatR) * (A * A / 2 + (5 - T + 9 * C + 4 * C * C) * ((A ** 4) / 24)) +
         (61 - 58 * T + T * T + 600 * C -
          330 * eccPrimeSquared) * ((A ** 6) / 720))
    return Easting, Northing


def to_list(obj):
    """Summary
    
    Args:
        obj (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        obj = [obj]
    else:
        obj = list(obj)
        # obj = [x for x in obj]
    return obj


def fileparts(filename):
    """Summary
    
    Args:
        filename (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    path = os.path.dirname(filename)
    filename = os.path.basename(filename)
    ext = os.path.splitext(filename)[1]
    return path, filename, ext


@list_or_numpy
def truncate(nums):
    """Summary
    
    Args:
        nums (TYPE): Description
    
    Returns:
        TYPE: Description
    
    Deleted Parameters:
        sig (int, optional): Description
    """
    sig = 5
    if nums is not None:
        if hasattr(nums, '__iter__'):
            retval = []
            for x in nums:
                if x == 0:
                    retval.append(0)
                else:
                    retval.append(round(x, -int(floor(log10(abs(x)))) + (sig - 1)))
            # retval = [round(x, -int(floor(log10(abs(x)))) + (sig - 1)) for x in nums]
            # print('This is a list')
        else:
            if nums == 0:
                retval = nums
            else:
                retval = round(nums, -int(floor(log10(abs(nums)))) + (sig - 1))
        return retval


def center_locs(locations):
    """Summary
    
    Args:
        locations (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    X, Y = locations[:, 0], locations[:, 1]
    center = (np.mean(X), np.mean(Y))
    X, Y = X - center[0], Y - center[1]
    return np.array([[x, y] for x, y in zip(X, Y)]), center


def rotate_locs(locs, azi=0):
    """Summary
    
    Args:
        locs (TYPE): Description
        azi (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    if azi == 0:
        return locs
    azi = np.deg2rad(azi)
    R = np.array([[np.cos(azi), -np.sin(azi)], [np.sin(azi), np.cos(azi)]])
    rot_locs = np.dot(locs, R.T)
    return rot_locs


def list2np(data):
    """Summary
    
    Args:
        data (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    try:
        return np.array(data)
    except ValueError as e:
        print('Error here')
        print(data)
        print('Error over')
        raise(e)


def np2list(data):
    """Summary
    
    Args:
        data (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [x for x in data]


def flatten_list(List):
    """Summary
    
    Args:
        List (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [float(point) for sublist in List for point in sublist]


def validate_input(inval, expected_type):
    """Summary
    
    Args:
        inval (TYPE): Description
        expected_type (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    try:
        retval = expected_type(inval)
    except ValueError:
        print('Conversion not possible. Please enter a valid input (of type {})'.format(expected_type))
        return False
    else:
        return retval


def compute_rho(site, calc_comp=None, errtype='none'):
    """Summary
    Computes apparent resistivity from impedance data for a given site.
    Args:
        site (Site): wsinv3dmt.data_structures Site object containing impedance data to be converted
        calc_comp (string, optional): Type of apparent resistivity to calculate.
        (Rxx, Rxy, Ryy, Ryx, Rdet {Default})
    
    Returns:
        np.array: Apparent resistivity of the chosen type
        np.array: Associated errors. Note that the errors are set to 0 for the determinant apparent
                  resistivities (Proper calculatation not implemented yet)
    """
    COMPS = ('RHOXX', 'RHOXY',
             'RHOYY', 'RHOYX',
             'RHODET')
    if calc_comp.upper() not in COMPS:
        print('{} not a valid option'.format(calc_comp))
        return
    mu = 0.2 / (4 * np.pi * 1e-7)
    if calc_comp is None:
        calc_comp = 'rhodet'
    if calc_comp.lower() == 'rhodet':
        det = compute_MT_determinant(site)
        rho_data = det * np.conj(det) * mu * site.periods
        rho_error = rho_log10Err = rho_data * 0
        # Do errors here...
    else:
        comp = ''.join(['Z', calc_comp[3:]])
        z = site.data[''.join([comp, 'R'])] -1j * site.data[''.join([comp, 'I'])]
        rho_data = z * np.conj(z) * mu * site.periods
        if errtype.lower() != 'none':
            zeR = getattr(site, errtype)[''.join([comp, 'R'])]
            zeI = getattr(site, errtype)[''.join([comp, 'I'])]
            if len(zeR) == 0:
                zeR = zeI = z * 0
            C = 2 * site.periods * mu
            rho_error = np.sqrt((C * np.real(z) * zeR) ** 2 + (C * np.imag(z) * zeI) ** 2)
            rho_log10Err = ( rho_error * np.sqrt(2) /
                             ((C * np.real(z) ** 2 + C * np.imag(z) ** 2) * np.log(10)) )
        else:
            rho_error = rho_log10Err = rho_data * 0
    return np.real(rho_data), np.real(rho_error), np.real(rho_log10Err)


def compute_phase(site, calc_comp=None, errtype='none'):
    COMPS = ('PHAXX', 'PHAXY',
             'PHAYY', 'PHAYX',
             'PHADET')
    if calc_comp.upper() not in COMPS:
        print('{} not a valid option'.format(calc_comp))
        return
    if calc_comp is None:
        calc_comp = 'phadet'
    if calc_comp.lower() == 'phadet':
        det = compute_MT_determinant(site)
        pha_data = np.angle(det, deg=True) + 90
        pha_error = pha_data * 0
    else:
        comp = ''.join(['Z', calc_comp[3:]])
        z = site.data[''.join([comp, 'R'])] -1j * site.data[''.join([comp, 'I'])]
        pha_data = np.angle(z, deg=True)
        if comp[1:] == 'YX':
            pha_data = 180 + pha_data
        pha_error = pha_data * 0
    return pha_data, pha_error


def compute_MT_determinant(site):
    try:
        det = np.sqrt((site.data['ZXYR'] - 1j * site.data['ZXYI']) *
                      (site.data['ZYXR'] - 1j * site.data['ZYXI']) -
                      (site.data['ZXXR'] - 1j * site.data['ZXXI']) *
                      (site.data['ZYYR'] - 1j * site.data['ZYYI']))
    except KeyError as e:
        print('Determinant cannot be computed unless all impedance components are available')
        raise e
    else:
        return det






