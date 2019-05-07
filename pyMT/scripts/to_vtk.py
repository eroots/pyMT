import pyMT.data_structures as WSDS
from pyMT.IO import verify_input
import os
from pyMT.WSExceptions import WSFileError
from pyMT.utils import project


def transform_locations(dataset, UTM):
    dataset.raw_data.locations = dataset.raw_data.get_locs(mode='latlong')
    UTM_number = int(UTM[:2])
    UTM_letter = UTM[-1]
    for ii in range(len(dataset.raw_data.locations)):
        easting, northing = project((dataset.raw_data.locations[ii, 1],
                                     dataset.raw_data.locations[ii, 0]),
                                    zone=UTM_number, letter=UTM_letter)[2:]
        dataset.raw_data.locations[ii, 1], dataset.raw_data.locations[ii, 0] = easting, northing


def to_vtk(outfile, datafile=None, listfile=None, modelfile=None,
           datpath=None, origin=None, UTM=None, sea_level=0, resolutionfile=None, transform_coords=None):
    if not outfile:
        print('Output file required!')
        return
    if listfile or datafile:
        dataset = WSDS.Dataset(modelfile=modelfile,
                               datafile=datafile,
                               listfile=listfile,
                               datpath=datpath)
    model = WSDS.Model(modelfile=modelfile)
    if not UTM:
        print('Using dummy UTM zone')
        UTM = '999'
    if transform_coords == 'y':
        print('Transforming locations to UTM {}'.format(UTM))
        transform_locations(dataset, UTM)

    if not origin and not listfile:
        print('You must either specify the origin, or the list file to calculate it from.')
        return
    elif (not origin and listfile) or transform_coords == 'y':
        origin = dataset.raw_data.origin
        print('Setting origin as {}'.format(origin))
    if modelfile:
        model.origin = origin
        model.UTM_zone = UTM
        print('Writing model to {}'.format('_'.join([outfile, 'model.vtk'])))
        if resolutionfile:
            print('Adding resolution')
            resolution = WSDS.Model(modelfile=resolutionfile)
            model.resolution = resolution.vals
        model.to_vtk(outfile, sea_level=sea_level)
    if listfile:
        print('Writing model to {}'.format('_'.join([outfile, 'sites.vtk'])))
        # dataset.raw_data.locations = dataset.raw_data.get_locs(mode='centered')
        dataset.raw_data.locations -= (origin[1], origin[0])
        dataset.raw_data.to_vtk(origin=origin, UTM=UTM, outfile=outfile, sea_level=sea_level)
    elif datafile:
        print('Writing model to {}'.format('_'.join([outfile, 'sites.vtk'])))
        dataset.data.to_vtk(origin=origin, UTM=UTM, outfile=outfile, sea_level=sea_level)


def get_inputs():
    raw_origin = '0, 0'

    args = {}
    # print('Output model, sites, or both? (m/d/b) {Default = b}')
    to_output = verify_input(message='Output model, sites, or both? (m/d/b)',
                             expected='mbd', default='b')
    if not to_output:
        to_output = 'b'
    if to_output == 'm' or to_output == 'b':
        # Get model name
        modelfile = verify_input('Input model name', expected='read')
        args.update({'modelfile': modelfile})
        include_resolution = verify_input('Include model resolution?',
                                          expected='yn', default='n')
        if include_resolution == 'y':
            resolution_file = verify_input('Resolution file name:', expected='read',
                                           default='Resolution0_inverted.model')
            args.update({'resolutionfile': resolution_file})
    if to_output == 'd' or to_output == 'b':
        # Get data or list file
        datafile = verify_input('Input data or list file name', expected='read')
        # If listfile, make sure the conversion to raw data works
        if '.lst' in datafile:
            path = os.path.abspath(datafile)
            datpath = input('Path to .dat files \n' +
                            '[Default: {}] > '.format(path))
            try:
                raw_data = WSDS.RawData(listfile=datafile, datpath=datpath)
                raw_origin = str(raw_data.origin).replace('(', '').replace(')', '')
                args.update({'datpath': datpath})
                args.update({'listfile': datafile})
                # args.update({'origin': origin})

            except WSFileError as err:
                print(err.message)
                # Otherwise just get the data file
        elif '.dat' in datafile:
            args.update({'datafile': datafile})
    origin = verify_input('Input model origin as Easting, Northing',
                          expected='numtuple',
                          default=raw_origin)
    args.update({'origin': origin})
    args.update({'UTM': verify_input('UTM Zone',
                                     expected=str,
                                     default='dummy')})
    if args['UTM'] != 'dummy':
        args.update({'transform_coords': verify_input('Do you want to transform the coordinates to this UTM zone?',
                                                      expected='yn',
                                                      default='y')})
    else:
        args.update({'transform_coords': 'n'})
    sea_level = verify_input('Sea level adjustment (Positive for above sea level)',
                             expected=float, default=0)
    args.update({'sea_level': sea_level})
    outfile = verify_input('Base output file name', expected='write')
    args.update({'outfile': outfile})

    return args


if __name__ == '__main__':
    try:
        args = get_inputs()
    except KeyboardInterrupt:
        print('\nQuitting...')
    else:
        # print(args)
        to_vtk(**args)
