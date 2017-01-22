import wsinv3dmt.data_structures as WSDS
import wsinv3dmt.IO as WSIO
import wsinv3dmt.utils as utils
import numpy as np
import sys
import matplotlib.pyplot as plt
from wsinv3dmt.WSExceptions import WSFileError


def greetings():
    print('\t Welcome to j2ws3d Version 2a\n\n')


def MTU_data_types(data, dType=''):
    MTU_5a = [-10400., -8800., -7200., -6000., -5200.,
              -4400., -3600., -3000, -2600., -2200., -1800.,
              -1500., -1300., -1100., -900., -780., -640., -530.,
              -460., -390., -360., -320., -312., -265., -229., -216.,
              -194., -180., -159., -132., - 115., -108., -97., -90., -79.,
              -66., -57., -54., -49., -45., -40., -39., -33., -27.5, -27.,
              -22.50, -19.50, -18.80, -16.50, -16.20, -13.70, -13.50,
              -11.20, -9.700, -9.40, -8.300, -8.100, -6.900,
              -6.800, -5.600, -4.900, -4.700, -4.100, -3.400, -2.810,
              -2.440, -2.340, -2.060, -1.720, -1.690, -1.410, -1.220,
              -1.170, -1.030, -1.020, 1.163, 1.190,
              1.429, 1.695, 1.961, 2.326, 2.857,
              3.413, 3.937, 4.651, 5.682, 6.849, 7.874, 9.346, 11.36,
              13.70, 15.87, 18.52, 22.73, 27.03, 31.25, 37.17, 45.45,
              54.64, 62.89, 74.63, 90.91, 108.7, 126.6, 149.3, 181.8,
              217.4, 250.0, 294.1, 363.6, 436.7, 505.1, 595.2, 729.9,
              877.2, 1010., 1190., 1449., 1754., 2000., 2381., 2941.]
    MTU_5 = [-320.0, -265.0, -229.0, -194.0, -159.0, -132.0, -115.0, -97.00,
             -79.00, -66.00, -57.00, -49.00, -40.00, -33.00, -27.50, -22.50,
             -18.80, -16.20, -13.70, -11.20, -9.400, -8.100, -6.900, -5.600,
             -4.700, -4.100, -3.400, -2.810, -2.340, -2.030, -1.720, -1.410,
             -1.170, -1.020, 1.163, 1.429, 1.695, 1.961, 2.326, 2.857,
             3.413, 3.937, 4.651, 5.682, 6.849, 7.874, 9.346, 11.36,
             13.70, 15.87, 18.52, 22.73, 27.03, 31.25, 37.17, 45.45,
             54.64, 62.89, 74.63, 90.91, 108.7, 126.6, 149.3, 181.8,
             217.4, 250.0, 294.1, 363.6, 436.7, 505.1, 595.2, 729.9,
             877.2, 1010., 1190., 1449., 1754., 2000., 2381., 2941.]
    if dType.lower() == '5':
        retval = MTU_5
    elif dType.lower() == 'a':
        retval = MTU_5a
    elif dType.lower() == 'b':
        retval = list(set(MTU_5).union(set(MTU_5a)))
    elif dType.lower() == 'o':
        retval = WSIO.read_freqset()
    elif dType.lower() == 'm':
        retval = data.master_periods.keys()
    elif dType.lower() == 'n':
        retval = data.narrow_periods.keys()
    else:
        print('Input {} not recognized, try again.'.format(dType))
        return False
    for ii, val in enumerate(retval):
        if val < 0:
            retval[ii] = -1 / val
    return retval


def verify_input(message, expected, default=None):
        while True:
            ret = input(' '.join([message, '[Default: {}] >'.format(default)]))
            if ret == '' and default is not None:
                ret = default
            if expected == 'read':
                if utils.check_file(ret):
                    return ret
                else:
                    print('File not found. Try again.')
            elif expected == 'write':
                if not utils.check_file(ret):
                    return ret
                else:
                    resp = verify_input('File exists, overwrite?', default='y', expected='yn')
                    if resp == 'y':
                        return ret
                    else:
                        return False
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


def main_data(args):
    defaults = {'azimuth': 0.0, 'flag_outliers': 'y', 'outlier_errmap': 10,
                'lowTol': 2.0, 'highTol': 10.0, 'use_TF': 'y', 'period_choice': 'n',
                'cTol': 0.5, 'hfreq_errmap': 20, 'XXYY_errmap': 10,
                'no_period_errmap': 50}
    if set(('dbg', 'debug', '-dbg', '-debug')).intersection(set([x.lower() for x in args])):
        DEBUG = True
        WSDS.DEBUG = True
    if set(('i', '-i')).intersection(set([x.lower() for x in args])):
        interactive = True
    interactive = True
    # Get list file
    list_file = verify_input('Enter list file name:', expected='read')
    try:
        site_names = WSIO.read_sites(list_file)
    except WSFileError as e:  # Error is raised if something is wrong with the file.
        print(e.message)
        print('Exiting...')
        return
    dataset = WSDS.Dataset(listfile=list_file)
    # Get desired azimuth
    azimuth = verify_input('Desired azimuth (deg. from True North)',
                           expected=float, default=defaults['azimuth'])
    flag_outliers = verify_input('Adjust error map for outliers?',
                                 default=defaults['flag_outliers'], expected='yn')
    if flag_outliers == 'y':
        dataset.data.OUTLIER_MAP = verify_input('Set outlier error map',
                                                default=defaults['outlier_errmap'], expected=int)
    else:
        dataset.data.OUTLIER_MAP = 1
    dataset.data.HIGHFREQ_MAP = verify_input('Error map on high frequencies (>1000Hz)',
                                             default=defaults['hfreq_errmap'], expected=int)
    dataset.data.XXYY_MAP = verify_input('XXYY error map',
                                         default=defaults['XXYY_errmap'],
                                         expected=int)
    dataset.data.NO_PERIOD_MAP = verify_input('Missing period error map',
                                              default=defaults['no_period_errmap'],
                                              expected=int)
    lowTol = verify_input('High frequency (>1Hz) matching tolerance %-age',
                          default=defaults['lowTol'], expected=float)
    lowTol /= 100
    highTol = verify_input('High frequency (>=10s) matching tolerance %-age',
                           default=defaults['highTol'], expected=float)
    highTol /= 100

    raw_data = dataset.raw_data
    num_TF = 0
    for site in raw_data.sites.values():
        if 'TZXR' in site.components:
            num_TF += 1
    print('{} sites out of {} have tipper data'.format(num_TF, len(site_names)))
    use_TF = verify_input('Would you like to include TF data?',
                          default=defaults['use_TF'], expected='yn')
    if use_TF:
        defaults.update({'inv_type': 5})
    else:
        defaults.update({'inv_type': 1})
    dType = verify_input('Which periods would you like to choose from?\n'
                         'Options are MTU-5, MTU-A, both, freqset, all, or program selected '
                         '(5/A/b/o/m/n)', default=defaults['period_choice'], expected='5aboan')
    # if dType == 'n':
    cTol = verify_input('Required fraction of sites containing each period?',
                        default=defaults['cTol'], expected=float)
    if cTol > 1:
        cTol /= 100
    period_set = raw_data.narrow_period_list(periods=MTU_data_types(data=raw_data, dType=dType),
                                             high_tol=highTol,
                                             low_tol=lowTol,
                                             count_tol=cTol)
    sorted_periods = sorted(period_set.keys())
    # Can set this up later to only run if -i is set (or set that as the default)
    # Want user to be able to decide between interactive (plotting) mode, and just
    # straight command line input.
    chosen = pick_periods(sorted_periods, period_set, interactive)
    inv_type = int(verify_input('Inversion Type?', default=defaults['inv_type'],
                                expected='12345'))
    # Still have to make sure this method also applies the error map properly.
    dataset.get_data_from_raw(lTol=lowTol, hTol=highTol, periods=chosen,
                              components=WSIO.get_components(invType=inv_type))
    if azimuth != 0:
        dataset.rotate_sites(azimuth)
    while True:
        outdata = verify_input('Output file name?', default='.data', expected='write')
        if outdata:
            if outdata[-5:] != '.data':
                outdata = ''.join([outdata, '.data'])
            dataset.write_data(outfile=outdata, overwrite=True)
            break
    return dataset


def pick_periods(sorted_periods, period_set, interactive):
    sorted_ratios = [period_set[p] * 100 for p in sorted_periods]
    log_per = [utils.truncate(np.log10(x)) for x in sorted_periods]
    num = list(range(1, len(sorted_periods) + 1))
    for ii, p in enumerate(sorted_periods):
        if p < 1:
            p = -1 / p
        sorted_periods[ii] = utils.truncate(p)
    data = np.transpose(np.array([num, sorted_periods, log_per, sorted_ratios]))
    if interactive and len(num) <= 45:
        col_labels = ("Number", "Period", "Log10(Period)", "Percentage of sites")
        fig, ax = plt.subplots(1, 1)
        ax.axis('off')
        ax.table(cellText=data,
                 colLabels=col_labels,
                 loc='center')
        plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.9])
        plt.ion()
        plt.show()
    else:
        print('Number\tPeriod\tLog10(Period)\t%-age of Sites\n')
        for idx in range(data.shape[0]):
            print('{:<8d}{:7.2f}{:13.2f}{:14.2f}'.format(int(data[idx, 0]),
                                                         data[idx, 1],
                                                         data[idx, 2],
                                                         data[idx, 3]))
    chosen = []
    print('Select periods by entering the corresponding integers. Enter 0 when done.')
    while True:
        chosen.append(verify_input('Enter period number', expected=int, default=0))
        if chosen[-1] == 0:
            print('Selected Periods:')
            for idx in chosen:
                print('{}\n'.format(sorted_periods[idx]))
            resp = verify_input('Continue with these?', default='y', expected='yn')
            if resp == 'y':
                break
            else:
                chosen = []
                print('Starting over...')
    return chosen


def main_mesh(args, data=None):
    if data is None:
        datafile = verify_input('Data file to use?', expected='read')


if __name__ == '__main__':
    greetings()
    try:
        if '-data' in sys.argv:
            data = main_data(sys.argv[1::])
        elif '-mesh' in sys.argv:
            mesh = main_mesh(sys.argv[1::])
        else:
            data = main_data(sys.argv[1:])
            mesh = main_mesh(sys.argv[1:], data=data)
    except KeyboardInterrupt:
        print('\nExiting...')
    except OSError:
        print('Wrong exit command, but exiting anyways...')
