import pyMT.data_structures as WSDS
import pyMT.utils as utils
import os
import re
from pyMT.WSExceptions import WSFileError


def to_vtk(outfile, datafile=None, listfile=None, modelfile=None,
           datpath=None, origin=None, UTM=None, sea_level=0):

    if not outfile:
        print('Output file required!')
        return
    dataset = WSDS.dataset(modelfile=modelfile,
                           datafile=datafile,
                           listfile=listfile,
                           datpath=datpath)
    if not origin and not listfile:
        print('You must either specify the origin, or the list file to calculate it from.')
        return
    elif not origin and listfile:
        origin = dataset.raw_data.origin()
        print('Setting origin as {}'.format(origin))
    if not UTM:
        print('Using dummy UTM zone')
        UTM = '999'
    if modelfile:
        dataset.model.origin = origin
        dataset.model.UTM_zone = UTM
        dataset.model.to_vtk(outfile, sea_level=sea_level)
    if listfile:
        dataset.raw_data.to_vtk(origin=origin, UTM=UTM, outfile=outfile, sea_level=sea_level)
    elif datafile:
        dataset.data.to_vtk(origin=origin, UTM=UTM, outfile=outfile, sea_level=sea_level)


def get_inputs():
    try:
        args = {}
        # print('Output model, sites, or both? (m/d/b) {Default = b}')
        to_output = input('Output model, sites, or both? (m/d/b) [Default = b] > ')
        if not to_output:
            to_output = 'b'
        if to_output == 'm' or to_output == 'b':
            # Get model name
            while True:
                modelfile = input('Input model name >  ')
                if utils.check_file(modelfile):
                    args.update({'modelfile': modelfile})
                    break
                else:
                    print('Model not found. Try again.')
        if to_output == 'd' or to_output == 'b':
            while True:
                # Get data or list file
                datafile = input('Input data or list file name > ')
                if utils.check_file(datafile):
                    # If listfile, make sure the conversion to raw data works
                    if '.lst' in datafile:
                        path = os.path.abspath(datafile)
                        datpath = input('Path to .dat files \n' +
                                        '[Default: {}] > '.format(path))
                        try:
                            raw_data = WSDS.RawData(listfile=datafile, datpath=datpath)
                            origin = raw_data.origin
                            print('Setting origin to {}'.format(origin))
                            args.update({'datpath': datpath})
                            args.update({'listfile': datafile})
                            args.update({'origin': origin})
                            break
                        except WSFileError as err:
                            print(err.message)
                    # Otherwise just get the data file
                    elif '.data' in datafile:
                        args.update({'datafile': datafile})
                        break
        while True:
            origin = input('Input model origin as Easting, Northing\n' +
                           '[Default: 0, 0] > ')
            if not origin:
                origin = '0, 0'
            try:
                origin = [float(x) for x in re.split(', | ', origin)]
            except ValueError:
                print('Could not convert {} to tuple'.format(origin))
            else:
                break

        args.update({'UTM': input('UTM Zone {Default: dummy}')})
        while True:
            try:
                sea_level = input('Sea level adjustment [Default: 0]')
                if sea_level == '':
                    sea_level = 0
                else:
                    sea_level = float(sea_level)
                break
            except ValueError:
                print('Could not convert {} to float'.format(sea_level))
        return args
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    try:
        args = get_inputs()
    except KeyboardInterrupt:
        print('Quitting...')
    else:
        print(args)
    # to_vtk(**args)
