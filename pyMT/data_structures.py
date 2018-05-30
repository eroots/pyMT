"""Summary
"""
import numpy as np
import os
from pyMT.WSExceptions import WSFileError
import pyMT.utils as utils
import pyMT.IO as WS_io


# ==========TODO=========== #
# Model implementation
# Response
# Methods for handling the data:
# Sort, get, filter out based on a criteria
# Mainly I would like stats on data per periods per site
# And be able to add a period/site to a Data object from
# another, or from a RawData object.
#
# Should be able to copy responses into data and vice versa, as well as copy parts
# of raw data into both.

# Need to figure out how to store and manipulate site location data. Rotating sites
# and making new grids, viewing rotated data should be easy and
# interactive so you can easily choose the best orientation for a given dataset.
# At some point I may have to modify the data structures to allow for other
# kinds of models/data (i.e. not just WS). This should be as simple as writing new read/write
# methods that are called instead of the WS ones if a certain flag is given. As long as the
# read method sends back the same info as the WS one, it should be fine.
#
# Alternating between datasets only works well when the sites are exactly the same.
# I'll have to decide if this is a bug or a feature... For instance, the site list doesn't
# update to the new list of sites.

# Have to figure out what I'm going to do about error flags (i.e. outliers, no data, XXYY).
# Should probably also put together at least a rudimentary plot of sites locations that is tied
# to the plotter. Actual model plotter will take a lot of work, but in the meantime it would
# be nice to be able to see where the sites are that I'm looking at. Also, may to be able to sort the
# sites by location.

# Look into 'Sphinx' for creating doctstrings to document the modules.
# Will also eventually have to write tests so that I can easily check any changes I make.
# ========================= #

# Important note on usage: For most of the class, I assume it is being used correctly.
# In that I mean there is very little error checking to make sure you aren't trying pass in
# bogus site data and whatnot. Some of this is probably dangerous, other parts less so. Will
# eventually have to go through it and check which parts need
# to be enforced and which ones are internal, and therefore (hopefully) will remain consistent.

DEBUG = False


class Dataset(object):
    data_types = ('raw_data', 'response', 'data')
    """
    Container for all WS related information, including model, data, error, and response.
    Methods:
        read_model(modelfile): Reads the model contained in 'modelfile'.
                               if no file is given, an empty model object is created.
        read_data(listfile, datafile): Reads the data in 'datafile', with the site names given in
                                       'listfile'. If no listfile is specified, and self.site_names
                                       is empty, sites are numbered.
        read_response(responsefile): Reads the response data given in 'responsefile'.
                                     Site names are determined the same as with read_data
        read_raw_data(listfile, datpath): Reads the raw data from the .dat files specified in
                                          'listfile'. The path to the .dat files
                                          may be given in 'datpath', otherwise
                                          the path is taken to be the same as that for 'listfile'.
    Attributes:
        data (Data Object): Description
        model (Model Object): Description
        raw_data (RawData Object): Description
        response (Response Object): Description
    """

    def __init__(self, modelfile='', listfile='', datafile='',
                 responsefile='', datpath=''):
        """Summary

        Args:
            modelfile (str, optional): Description
            listfile (str, optional): Description
            datafile (str, optional): Description
            responsefile (str, optional): Description
            datpath (str, optional): Description
        """
        self.data = Data(listfile=listfile, datafile=datafile)
        self.model = Model(modelfile=modelfile)
        self.raw_data = RawData(listfile=listfile, datpath=datpath)
        self.response = Data(listfile=listfile, datafile=responsefile)
        self.rms = self.calculate_RMS()
        if not self.data.site_names:
            self.get_data_from_raw()
        azi = []
        num_dTypes = 0
        for dType in self.data_types:
            if self.has_dType(dType):
                azi.append(getattr(self, dType).azimuth)
                num_dTypes += 1
        if len(set(azi)) == 1 and len(azi) == num_dTypes:
            self.azimuth = azi[0]
        else:
            print('Not all data azimuths are equal. Defaulting to that set in the data file.')
            print('data: {}, raw_data: {}, response: {}'.format(self.data.azimuth,
                                                                self.raw_data.azimuth,
                                                                self.response.azimuth))
            self.rotate_sites(azi=self.data.azimuth)
            self.azimuth = self.data.azimuth
        if self.raw_data.initialized:
            try:
                self.freqset = WS_io.read_freqset(self.raw_data.datpath)
            except FileNotFoundError:
                print('Freqset not found')
                self.freqset = None
        else:
            self.freqset = None

    def has_dType(self, dType):
        if dType in self.data_types:
            return bool(getattr(self, dType).sites)
        else:
            print('{} is not a valid data type'.format(dType))
            return False

    @utils.enforce_input(sites=list, periods=list, components=list, hTol=float, lTol=float)
    def get_data_from_raw(self, lTol=None, hTol=None, sites=None, periods=None, components=None):
        # if components:
        #     if any(comp not in Data.ACCEPTED_COMPONENTS for comp in components):

        #         print('Invalid component passed to keyword "components"')
        #         return
        self.data.sites = self.raw_data.get_data(sites=sites, periods=periods,
                                                 components=components, lTol=lTol,
                                                 hTol=hTol)
        self.data.site_names = [site for site in self.raw_data.site_names]
        for site in self.data.sites.keys():
            self.data.sites[site].detect_outliers(self.data.OUTLIER_MAP)
        self.data._runErrors = []
        self.data.periods = np.array([p for p in self.data.sites[self.data.site_names[0]].periods])
        self.data.components = Data.ACCEPTED_COMPONENTS
        self.data.locations = self.data.get_locs()
        self.data.center_locs()
        self.data.azimuth = 0  # Azi is set to 0 when reading raw data, so this will be too.

    def read_model(self, modelfile=''):
        """Summary

        Args:
            modelfile (str, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            self.model.__read__(modelfile)
        except WSFileError as e:
            # self.model = Model()
            print(e.message)

    def read_data(self, listfile='', datafile=''):
        """Summary

        Args:
            listfile (str, optional): Description
            datafile (str, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            self.data.__read__(listfile=listfile, datafile=datafile)
        except WSFileError as e:
            # self.data = Data()
            print(e.message)

    def write_list(self, outfile='', overwrite=False):
        if not outfile:
            print('You should probably name your output file...')
            return False
        if not utils.check_file(outfile) or overwrite:
            self.data.write_list(outfile)
            return True
        else:
            print('File already exists')
            return False

    def write_data(self, outfile='', overwrite=False, file_format='wsinv3dmt'):
        if outfile == '':
            print('You should probably name your output first...')
            return False
        if not utils.check_file(outfile) or overwrite:
            self.data.write(outfile=outfile, file_format=file_format)
            return True
        else:
            print('File already exists')
            return False

    def to_vtk(self, origin=None, UTM=None, outfile=None, sea_level=0):
        if origin is None:
            if self.model.origin == (0, 0):
                origin = self.raw_data.origin
            else:
                origin = self.model.origin
        if self.data.sites:
            try:
                assert all(np.isclose(self.data.locations[:, 0] + origin[1],
                                      self.raw_data.locations[:, 0]))
                assert all(np.isclose(self.data.locations[:, 1] + origin[0],
                                      self.raw_data.locations[:, 1]))
            except AssertionError:
                print('Data site locations do not match raw data site locations!')
                return
        if not UTM and not self.model.UTM_zone:
            print('UTM zone must be set to something!')
            return
        elif not UTM:
            UTM = self.model.UTM_zone
        if not outfile:
            print('Need to specify outfile!')
            return
        outfile = os.path.splitext(outfile)[0]
        mod_outfile = ''.join([outfile, '_model.vtk'])
        site_outfile = ''.join([outfile, '_sites.vtk'])
        self.model.to_vtk(outfile=mod_outfile, origin=origin, UTM=UTM, sea_level=sea_level)
        self.data.to_vtk(outfile=site_outfile, UTM=UTM, origin=origin, sea_level=sea_level)

    def read_raw_data(self, listfile='', datpath=''):
        """Summary

        Args:
            listfile (str, optional): Description
            datpath (str, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            self.raw_data.__read__(listfile=listfile, datpath=datpath)
        except WSFileError as e:
            # self.rawData = RawData()
            print(e.message)

    def read_response(self, respfile=''):
        """Summary

        Args:
            respfile (str, optional): Description

        Returns:
            TYPE: Description
        """
        try:
            self.data.__read__(respfile)
        except WSFileError as e:
            # self.response = Response()
            print(e.message)

    @utils.enforce_input(site_names=list, dTypes=list)
    def get_sites(self, site_names, dTypes='all'):
        site_list = {'raw_data': [], 'data': [], 'response': []}
        if len(dTypes) == 1:  # or isinstance(dTypes, str):
            if dTypes[0].lower() == 'all':
                dTypes = ['raw_data', 'response', 'data']
            # else:
                # dTypes = utils.to_list(dTypes)
        for site in site_names:
            for dType in dTypes:
                if dType and getattr(self, dType).sites:
                    site_list[dType].append(getattr(self, dType).sites[site])
        return site_list

    def rotate_sites(self, azi=0):
        self.azimuth = azi
        if self.has_dType('data'):
            self.data.rotate_sites(azi=azi)
            assert (self.azimuth == self.data.azimuth)
        if self.has_dType('raw_data'):
            self.raw_data.rotate_sites(azi=azi)
            assert (self.azimuth == self.raw_data.azimuth)
        if self.has_dType('response'):
            self.response.rotate_sites(azi=azi)
            assert (self.azimuth == self.response.azimuth)

    def remove_sites(self, sites):
        """
        Removes the specified site from the 'data' and 'response' attributes.
        Doesn't remove from 'raw_data' as the site might be added back in later.
        """
        if self.has_dType('data'):
            self.data.remove_sites(sites=sites)
        if self.has_dType('response'):
            self.response.remove_sites(sites=sites)
        if self.has_dType('raw_data'):
            self.raw_data.remove_sites(sites=sites)

    def add_site(self, site):
        if self.has_dType('data'):
            # Need to make sure that the added data is consistent with current, if not, get
            # the missing data from the relevant raw_data site
            self.data.add_site(site['data'])
        if self.has_dType('response'):
            self.response.add_site(site['response'])
        if self.has_dType('raw_data'):
            self.raw_data.add_site(site['raw_data'])

    def sort_sites(self, order=None):
        if order in (None, 'Default'):
            self.data.site_names = [site for site in self.raw_data.site_names
                                    if site in self.data.site_names]
        elif order.lower() == 'west-east':
            self.data.site_names = sorted(self.data.site_names,
                                          key=lambda x: self.data.sites[x].locations['Y'])
        elif order.lower() == 'south-north':
            self.data.site_names = sorted(self.data.site_names,
                                          key=lambda x: self.data.sites[x].locations['X'])
        elif order.lower() == 'clustering':
            sites = sorted(self.data.site_names,
                           key=lambda x: (self.data.sites[x].locations['X'],
                                          self.data.sites[x].locations['Y']))
            self.data.site_names = sites
        else:
            print('Order {} not recognized.'.format(order))
            return
        self.data.locations = self.data.get_locs()

    def regulate_errors(self, multiplier=2.5, fwidth=1):
        use_log = False
        for site in self.raw_data.site_names:
            raw_site = self.raw_data.sites[site]
            data_site = self.data.sites[site]
            for comp in self.data.sites[site].components:
                if comp[0].lower() == 'z':
                    error_map = np.zeros(data_site.data[comp].shape)
                    smoothed_data = utils.geotools_filter(np.log10(raw_site.periods),
                                                          np.sqrt(raw_site.periods) * raw_site.data[comp],
                                                          fwidth=fwidth, use_log=use_log)
                    for ii, p in enumerate(data_site.periods):
                        ind = np.argmin(abs(raw_site.periods - p))
                        max_error = multiplier * abs(np.sqrt(p) * data_site.data[comp][ii] -
                                                     smoothed_data[ind])
                        error_map[ii] = min([data_site.errmap[comp][ii],
                                             np.ceil(max_error / (np.sqrt(p) * data_site.errors[comp][ii]))])
                    self.data.sites[site].errmap[comp] = error_map
            self.data.sites[site].apply_error_floor()

    def calculate_RMS(self):
        NP = self.data.NP
        NS = self.data.NS
        NR = self.data.NR
        components = self.data.components
        if NS == 0 or NR == 0 or NP == 0:
            return None
        if NP != self.response.NP:
            print('Data and response periods do not match')
            print('Data NP: {}, Response NP: {}'.format(self.data.NP, self.response.NP))
            return None
        if NS != self.response.NS:
            print('Number of sites in data and response not equal')
            return None
        sites = self.data.site_names
        misfit = {site: {comp: np.zeros((NP)) for comp in components}
                  for site in sites}
        # period_misfit = {site: np.zeros(NP) for site in sites}
        period_misfit = {comp: np.zeros((NP)) for comp in components + ['Total']}
        comp_misfit = {comp: 0 for comp in components + ['Total']}
        total_misfit = 0
        for site in sites:
            all_misfits = utils.calculate_misfit(self.data.sites[site],
                                                 self.response.sites[site])
            misfit[site] = all_misfits[0]
            # period_misfit += all_misfits[1]
            total_misfit += np.mean(all_misfits[1]['Total'])
            for comp in comp_misfit.keys():
                period_misfit[comp] += all_misfits[1][comp]
                # comp_misfit[comp] += np.mean(period_misfit[comp])
        period_RMS = {comp: np.sqrt(period_misfit[comp] / NS) for comp in period_misfit.keys()}
        total_RMS = np.sqrt(total_misfit / NS)
        comp_RMS = {comp: np.sqrt(np.mean(period_misfit[comp]) / NS) for comp in comp_misfit.keys()}
        station_RMS = {site: {comp: np.sqrt(np.mean(misfit[site][comp])) for comp in components}
                       for site in sites}
        for site in sites:
            station_RMS[site].update({'Total': 0})
            for comp in components:
                station_RMS[site]['Total'] += station_RMS[site][comp] ** 2
            station_RMS[site]['Total'] = np.sqrt(station_RMS[site]['Total'] / NR)

        # return {'Total': total_RMS, 'Period': period_RMS,
        #         'Component': comp_RMS, 'Station': station_RMS}
        return {'Total': total_RMS, 'Component': comp_RMS,
                'Station': station_RMS, 'Period': period_RMS}
        # for site in sites:
        #     for comp in components:
        #         station_RMS[site][comp] = np.sqrt(np.mean(misfit[site][comp]))


class Data(object):
    """Container for data used in a WS inversion, including data points, errors, and error maps.

    Attributes:
        ACCEPTED_COMPONENTS (TYPE): Description
        components (TYPE): Description
        datafile (TYPE): Description
        periods (list): Description
        site_names (list): Description
        sites (dict): Description
        datafile - The file used to generate the data contained in the instance.
        site_names - The names of the sites contained in the instance.
        sites - a list of Site objects, each containing the data for that site.

    """
    ACCEPTED_COMPONENTS = ('ZXXR', 'ZXXI',
                           'ZXYR', 'ZXYI',
                           'ZYXR', 'ZYXI',
                           'ZYYR', 'ZYYI',
                           'TZXR', 'TZXI',
                           'TZYR', 'TZYI')
    IMPLEMENTED_FORMATS = {'MARE2DEM': '.emdata',
                           'WSINV3DMT': ['.data', '.resp'],
                           'ModEM': '.dat'}
    INVERSION_TYPES = {1: ('ZXXR', 'ZXXI',
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
                       6: ('RhoZXX', 'PhszXX',
                           'RhoZXY', 'PhszXY',
                           'RhoZYX', 'PhszYX',
                           'RhoZYY', 'PhszYY'),
                       7: ('RhoZXY', 'PhsZXY',
                           'RhoZYX', 'PhsZYX'),
                       8: ('TZYR', 'TZYI'),
                       9: ('RhoZXY', 'PhsZXY',
                           'RhoZYX', 'PhsZYX',
                           'TZYR', 'TZYI'),
                       10: ('RhoZXX', 'PhsZXX',
                            'RhoZXY', 'PhsZXY',
                            'RhoZYX', 'PhsZYX',
                            'RhoZYY', 'PhsZYY',
                            'TZYR', 'TZYI'),
                       11: ('log10RhoZXX', 'PhsZXX',
                            'log10RhoXY', 'PhsXY',
                            'log10RhoYX', 'PhsYX',
                            'log10RhoYY', 'PhsYY'),
                       12: ('log10RhoZXY', 'PhsZXY',
                            'log10RhoZYX', 'PhsZYX'),
                       13: ('log10RhoZXY', 'PhsZXY',
                            'log10RhoZYX', 'PhsZYX',
                            'TZYR', 'TZYI'),
                       14: ('log10RhoZXX', 'PhsZXX',
                            'log10RhoZXY', 'PhsZXY',
                            'log10RhoZYX', 'PhsZYX',
                            'log10RhoZYY', 'PhsZYY',
                            'TZYR', 'TZYI')}

    # Scale error map by how much the period differs, or do a straight error mapping
    # First value is 'scaled' or 'straight', second value is mult. For scaled, this is multiplied by
    # the percent difference between periods. I.E. if the chosen period is 1% away from an existing period,
    # and the map is 10, the error map is 1 * 10 = 10.
    HIGHFREQ_MAP = 20
    OUTLIER_MAP = 10
    XXYY_MAP = 5
    NO_PERIOD_MAP = 50
    NO_COMP_MAP = 9999

    def __init__(self, datafile='', listfile='', file_format=None):
        """Summary

        Args:
            datafile (str, optional): Path to the data file to be read in
            listfile (str, optional): Path to the file containing the station names,
                                      in the same order as given in the data file.
            file_format (str, optional): Format of the data file to be read in.
                                         Options are MARE2DEM, WSINV3DMT, ModEM. If not present,
                                         format will be inferred from data file extention.
        """
        self.datafile = datafile
        self.site_names = []
        self._runErrors = []
        self.periods = []
        self.inv_type = None
        self.azimuth = 0
        self.origin = None
        self.UTM_zone = None
        if not file_format or file_format in Data.IMPLEMENTED_FORMATS.keys():
            self.file_format = file_format
        else:
            print('File format {} not recognized. ' +
                  'Will attempt to determine file format from data file'.format(file_format))
            self.file_format = None

        self.__read__(listfile=listfile, datafile=datafile)
        if self.sites:
            self.components = self.sites[self.site_names[0]].components
            # self.locations = {site.name: site.locations for site in self.sites.values()}
            self.locations = self.get_locs()
            if not self.inv_type:
                self.auto_set_inv_type()
                print('Automatically setting inverison type.')
                print('Inverison type set to {}. Please ensure this is correct.'.format(self.inv_type))
        else:
            self.components = []
            self.locations = []

        # Decided against automatically inserting dummy TF data.
        # for site in self.sites.values():
        #     for comp in self.components:
        #         dummy_data = site.data[self.components[0]] * 1e-10
        #         if comp not in self.components:
        #             site.add_component(data={comp: dummy_data},
        #                                errors=1000,
        #                                errmap=site.NO_COMP_MAP)

    def __read__(self, listfile='', datafile='', invType=None):
        """Summary

        Args:
            listfile (str, optional): Description
            datafile (str, optional): Description
            invType (None, optional): Description

        Returns:
            TYPE: Description
        """
        # path, filename, ext = fileparts(datafile)

        if listfile and datafile:
            self.site_names = WS_io.read_sites(listfile)
        if datafile:
            # Might have to watch that if site names are read after data, that
            # The data site names are updated appropriately.
            ext = os.path.splitext(datafile)[1]
            if not self.file_format:
                if ext == '.emdata' or ext == '.resp':
                    self.file_format = 'MARE2DEM'
                elif ext == '.dat':
                    self.file_format = 'ModEM'
                elif ext == '.data' or 'resp' in datafile:
                    self.file_format = 'WSINV3DMT'
                else:
                    message = ('Acceptable formats are <Inversion Code> - <File Extension>:\n')
                    for code, ext in Data.IMPLEMENTED_FORMATS.items():
                        message += '{} => {}\n'.format(code, ext)
                    raise WSFileError(ID='fmt', offender=datafile, extra=message)
            all_data, other_info = WS_io.read_data(datafile=datafile,
                                                   site_names=self.site_names,
                                                   file_format=self.file_format, invType=invType)
            self.site_names = other_info['site_names']
            self.inv_type = other_info['inversion_type']
            self.origin = other_info['origin']
            self.UTM_zone = other_info['UTM_zone']
            self.sites = {}

            for site_name, site in all_data.items():
                self.sites.update({site_name:
                                   Site(name=site_name,
                                        data=site['data'],
                                        errors=site['errors'],
                                        errmap=site.get('errmap', None),
                                        periods=site['periods'],
                                        locations=site['locations'],
                                        azimuth=site['azimuth'],
                                        errfloorZ=site.get('errFloorZ', 0),
                                        errfloorT=site.get('errFloorT', 0),
                                        solve_static=site.get('SolveStatic', 0)
                                        )})
            # if not self.site_names:
            #     self.site_names = [site for site in self.sites.keys()]
            self.periods = self.sites[self.site_names[0]].periods
            self.azimuth = self.sites[self.site_names[0]].azimuth
        else:
            self.sites = {}
        if invType:
            self.inv_type = invType

    def auto_set_inv_type(self):

        if set(self.used_components) == set(Data.INVERSION_TYPES[1]):
            #  Full Impedance
            self.inv_type = 1

        elif set(self.used_components) == set(Data.INVERSION_TYPES[2]):
            #  Off-diagonal Impedance
            self.inv_type = 2

        elif set(self.used_components) == set(Data.INVERSION_TYPES[3]):
            #  Full Tipper
            self.inv_type = 3

        elif set(self.used_components) == set(Data.INVERSION_TYPES[4]):
            #  Off-Diagonal + Full Tipper
            self.inv_type = 4

        elif set(self.used_components) == set(Data.INVERSION_TYPES[5]):
            #  Full Impedance + Full Tipper
            self.inv_type = 5

        elif set(self.used_components) == set(Data.INVERSION_TYPES[6]):
            #  Full Rho + Phase
            self.inv_type = 6

        elif set(self.used_components) == set(Data.INVERSION_TYPES[7]):
            #  Off-Diagonal Rho + Phase
            self.inv_type = 7

        elif set(self.used_components) == set(Data.INVERSION_TYPES[8]):
            #  2-D Tip
            self.inv_type = 8

        elif set(self.used_components) == set(Data.INVERSION_TYPES[9]):
            #  Off-Diagonal Rho + Phase + 2-D Tipper
            self.inv_type = 9

        elif set(self.used_components) == set(Data.INVERSION_TYPES[10]):
            #  Full Rho + Phase + 2-D Tipper
            self.inv_type = 10

        elif set(self.used_components) == set(Data.INVERSION_TYPES[11]):
            #  Full log10Rho + Phase
            self.inv_type = 11

        elif set(self.used_components) == set(Data.INVERSION_TYPES[12]):
            #  Off-Diagonal log10Rho + Phase
            self.inv_type = 12

        elif set(self.used_components) == set(Data.INVERSION_TYPES[13]):
            #  Off-Diagonal log10Rho + Phase + 2-D Tipper
            self.inv_type = 13

        elif set(self.used_components) == set(Data.INVERSION_TYPES[14]):
            #  Full Rho + Phase + 2-D Tipper
            self.inv_type = 14

    def reset_errors(self, error_floor=None, components=None, sites=None):
        if not sites:
            sites = self.site_names
        elif isinstance(sites, str):
            sites = [sites]
        for site in sites:
            self.sites[site].reset_errors(error_floor=error_floor, components=components)

    def add_noise(self, noise_level=5, components=None, sites=None):
        if not sites:
            sites = self.site_names
        elif isinstance(sites, str):
            sites = [sites]
        for site in sites:
            self.sites[site].add_noise(noise_level=noise_level, components=components)

    def set_error_map(self):
        for site in self.site_names:
            # Make sure the error map is fresh first
            self.sites[site].errmap = self.sites[site].generate_errmap(mult=1)
        self.apply_XXYY_map()
        self.apply_hfreq_map()
        self.apply_outlier_map()
        self.apply_no_data_map()

    def apply_no_data_map(self):
        for site in self.site_names:
            for comp in self.components:
                for ii, flag in enumerate(self.sites[site].flags[comp]):
                    if flag == self.sites[site].NO_COMP_FLAG:
                        self.sites[site].errmap[comp][ii] *= self.NO_COMP_MAP
                    elif flag == self.sites[site].NO_PERIOD_FLAG:
                        self.sites[site].errmap[comp][ii] *= self.NO_PERIOD_MAP

    def apply_outlier_map(self):
        for site in self.site_names:
            self.sites[site].detect_outliers(self.OUTLIER_MAP)

    def apply_XXYY_map(self):
        for site in self.site_names:
            for comp in self.components:
                    if 'xx' in comp.lower() or 'yy' in comp.lower():
                        self.sites[site].errmap[comp] *= self.XXYY_MAP

    def apply_hfreq_map(self):
        idx = np.where(self.periods <= 1 / 1000)
        for site in self.site_names:
            for comp in self.components:
                self.sites[site].errmap[comp][idx] *= self.HIGHFREQ_MAP

    @property
    def NP(self):
        return len(self.periods)

    @property
    def NS(self):
        return len(self.sites)

    @property
    def NR(self):
        return len(self.used_components)

    @property
    def used_components(self):
        if self.inv_type is None:
            # print('Inversion Type not set. Returning currently set components')
            components = self.components
        else:
            components = Data.INVERSION_TYPES[self.inv_type]
        # elif self.inv_type == 1:
        #     components = self.ACCEPTED_COMPONENTS[:8]
        # elif self.inv_type == 2:
        #     components = self.ACCEPTED_COMPONENTS[2:5]
        # elif self.inv_type == 3:
        #     components = self.ACCEPTED_COMPONENTS[8:]
        # elif self.inv_type == 4:
        #     components = self.ACCEPTED_COMPONENTS[2:]
        # elif self.inv_type == 5:
        #     components = self.ACCEPTED_COMPONENTS
        return components

    @property
    def inv_type(self):
        return self._inv_type

    @inv_type.setter
    def inv_type(self, val):
        if val and val not in Data.INVERSION_TYPES.keys():
            raise Exception('{} is not a valid inversion type'.format(val))
        self._inv_type = val

    def write(self, outfile, to_write=None, file_format=None):
        if not file_format:
            file_format = self.file_format
        WS_io.write_data(data=self, outfile=outfile, to_write=to_write, file_format=file_format)

    def write_list(self, outfile):
        WS_io.write_list(data=self, outfile=outfile)

    def to_vtk(self, outfile, UTM, origin=None, sea_level=0):
        if not origin:
            print('Using origin = (0, 0)')
            origin = (0, 0)
        WS_io.sites_to_vtk(self, outfile=outfile, origin=origin, UTM=UTM, sea_level=sea_level)

    def rotate_sites(self, azi):
        if DEBUG:
            print('Rotating site locations and data')
        if azi != self.azimuth:
            if azi < 0:
                azi = 360 + azi
            locs = self.get_locs()
            theta = self.azimuth - azi
            self.locations = utils.rotate_locs(locs, theta)
            for site_name in self.site_names:
                self.sites[site_name].rotate_data(azi)
            self.azimuth = azi

    def center_locs(self):
        self.locations, self._center = utils.center_locs(self.locations)
        if DEBUG:
            print('Centering site locations')
            for ii, site in enumerate(self.sites.values()):
                print('Site: {}, X: {}, Y: {}'.format(site.name,
                                                      self.locations[ii, 0],
                                                      self.locations[ii, 1]))

    def get_locs(self, site_list=None, azi=0):
        if site_list is None:
            site_list = self.site_names
        locs = np.array([[self.sites[name].locations['X'],
                          self.sites[name].locations['Y']]
                         for name in site_list])
        if azi != 0:
            locs = utils.rotate_locs(locs, azi)
        return locs

    def add_site(self, site):
        """
        Adds a site object to the data dictionary. It also checks to make sure that the
        periods and components it adds are the same as the existing sites. If not, it
        attempts to trim off the unnecessary parts. If it still doesn't conform, it won't
        add the site.

        Args:
            site (TYPE): Description
        """
        if self.sites:
            site0 = self.sites[self.site_names[0]]
            if site0.loosely_equal(site):
                self.sites.update({site.name: site})
                self.site_names.append(site.name)
            elif set(site0.periods).issubset(set(site.periods)) and \
                    set(site0.components).issubset(site.components):
                rmcomp = list(set(site.components) - set(site0.components))
                site.remove_component(rmcomp)
                rmperiods = list(set(site0.periods) - set(site.periods))
                site.remove_period(rmperiods)
                self.add_site(site)
            else:
                print('Site not added, as the data periods and/or components do not match.')
        else:  # Adding the first site to the data instance
            self.sites.update({site.name: site})
            self.site_names.append(site.name)

    @utils.enforce_input(sites=list)
    def remove_sites(self, sites):
        """Summary

        Args:
            site (TYPE): Description

        Returns:
            TYPE: Description
        """
        if len(self.sites.keys()) == 0:
            print('This data instance is already empty...')
            return
        for site in sites:
            try:
                name = site.name
            except AttributeError:
                name = site
            try:
                del self.sites[name]
                self.site_names.remove(name)
                self.locations = self.get_locs()
            except KeyError as e:
                print('Site {} does not exist'.format(name))
        return self

    def add_periods(self, sites):
        """Summary

        Args:
            sites (dict): A dictionary of {Name: Site object} where
            each site object contains the periods and data to be added for that site

        Returns:
            TYPE: Description
        """

        for site in self.sites.values():
            self.sites[site.name].add_periods(site=sites[site.name])
        self.periods = self.sites[self.site_names[0]].periods

    @utils.enforce_input(periods=list)
    def remove_periods(self, periods):
        """Summary

        Args:
            periods (TYPE): Description

        Returns:
            TYPE: Description
        """
        for site in self.site_names:
            for p in periods:
                self.periods = np.delete(self.periods, np.where(self.periods == p))
            self.sites[site].remove_periods(periods=periods)

    def add_component(self, data):
        """Summary

        Args:
            data (TYPE): Description

        Returns:
            TYPE: Description
        """
        raise(NotImplementedError)

    def remove_component(self, component):
        """Summary

        Args:
            component (TYPE): Description

        Returns:
            TYPE: Description
        """
        raise(NotImplementedError)

    def _check_consistency(self):
        """Summary

        Returns:
            TYPE: Description
        """
        name_check = set(self.site_names) == set([site_name for site_name in self.sites.keys()])
        comp_check = set(self.components) == set([comp for site in self.sites.values()
                                                  for comp in site.components])
        site_pers = [[period for period in site.periods] for site in self.sites.values()]
        site_per_check = all([np.equal(site_pers[0], p).all() for p in site_pers])
        site_comps = [[comp for comp in site.components] for site in self.sites.values()]
        site_comp_check = all([set(site_comps[0]) == set(comps) for comps in site_comps])
        return name_check, comp_check, site_per_check, site_comp_check

    def check_compromised_data(self, threshold=0.75):
        positive = ('ZXYR', 'ZYXI')
        negative = ('ZXYI', 'ZYXR')
        flagged_sites = {}
        for site in self.sites.values():
            comps = {'ZXYR': False, 'ZXYI': False,
                     'ZYXR': False, 'ZYXI': False}
            NP = len(site.periods)
            for comp in positive:
                if ((sum(site.data[comp] < 0) / NP) > threshold):
                    comps[comp] = True
            for comp in negative:
                if ((sum(site.data[comp] > 0) / NP) > threshold):
                    comps[comp] = True
            if any(comps.values()):
                flagged_comps = [comp for (comp, flag) in comps.items() if flag]
                flagged_sites.update({site.name: flagged_comps})
                print('Flagging site {}'.format(site.name))
                print('Inverted data at {}'.format(flagged_comps))
        return flagged_sites


class Model(object):
    """Summary
    """

    def __init__(self, modelfile='', data=None):
        self._xCS = []
        self._yCS = []
        self._zCS = []
        self._dx = []
        self._dy = []
        self._dz = []
        self.vals = []
        self.background_resistivity = 10000
        self.resolution = []
        self.file = modelfile
        self.origin = (0, 0)
        self.UTM_zone = None
        self.coord_system = 'local'
        if modelfile:
            self.__read__(modelfile=modelfile)
        elif data:
            self.generate_dummy_model()
            x_size = (np.max(data.locations[:, 1]) - np.min(data.locations[:, 1])) / 60
            y_size = (np.max(data.locations[:, 0]) - np.min(data.locations[:, 0])) / 60
            x_size = np.max((x_size, 1000))
            y_size = np.max((y_size, 1000))
            self.generate_mesh(site_locs=data.locations, regular=True, min_x=x_size, min_y=y_size,
                               num_pads=5, pad_mult=1.2)
            self.generate_zmesh(min_z=1, max_z=500000, NZ=60)

    def generate_dummy_model(self):
            self._xCS = [1] * 60
            self._yCS = [1] * 60
            self._zCS = [1] * 60
            self.vals = np.zeros((60, 60, 60)) + self.background_resistivity
            self.xCS = [1] * 60
            self.yCS = [1] * 60
            self.zCS = [1] * 60

    def __read__(self, modelfile=''):
        if modelfile:
            mod = WS_io.read_model(modelfile=modelfile)
            # Set up these first so update_vals isn't called
            self._xCS = mod['xCS']
            self._yCS = mod['yCS']
            self._zCS = mod['zCS']
            self.vals = mod['vals']
            self.xCS = mod['xCS']
            self.yCS = mod['yCS']
            self.zCS = mod['zCS']

    def to_vtk(self, outfile=None, sea_level=0, origin=None, UTM=None):
        if origin:
            self.origin = origin
        if UTM:
            self.UTM_zone = UTM
        if not outfile:
            print('Must specify output file name')
            return
        WS_io.model_to_vtk(self, outfile=outfile, sea_level=sea_level)

    def to_UTM(self, origin=None):
        '''
            Convert model coordinates to UTM
            Usage: model.to_UTM(origin=None) where origin is (Easting, Northing)
        '''
        if self.coord_system == 'local':
            if origin:
                self.origin = origin
            elif self.origin is None:
                print('Must specify origin if model.origin is not set')
                return False
            else:
                self._dx = [x + self.origin[1] for x in self._dx]
                self._dy = [y + self.origin[0] for y in self._dy]
        elif self.coord_system == 'latlong':
            self._dy, self._dx = utils.project((self._dx, self._dy))
        elif self.coord_system == 'UTM':
            print('Already in UTM')
            return
        self.coord_system = 'UTM'

    def to_latlong(self, zone=None):
        if self.coord_system == 'latlong':
            print('Already in latlong')
            return
        if self.coord_system == 'local':
            print('Trying to convert from local to UTM first...')
            ret = self.to_UTM()
            if not ret:
                print('Transformation not possible. Convert to UTM first.')
                return
        if not self.UTM_zone:
            self.UTM_zone = zone
        if not self.UTM_zone:
            print('UTM zone must be set or given')
            return
        if len(zone) == 2:
            number = int(zone[0])
        else:
            number = int(zone[:2])
        letter = zone[-1]
        lon = utils.unproject(number, letter,
                              self._dy,
                              [self.dx[round(len(self.dx) / 2)] for iy in self.dy])[0]
        lat = utils.unproject(number, letter,
                              [self.dy[round(len(self.dy) / 2)] for ix in self.dx],
                              self._dx)[1]
        self._dx, self._dy = (lat, lon)
        self.coord_system = 'latlong'
        return (lat, lon)

    def is_half_space(self):
        return np.all(np.equal(self.vals.flatten(), self.vals[0, 0, 0]))

    def update_vals(self, new_vals=None, axis=None, index=None, mode=None):
        if self.is_half_space():
            bg = self.vals[0, 0, 0]
            print('Changing values')
            self.vals = np.ones([self.nx, self.ny, self.nz]) * bg
        else:
            if index is not None and axis is not None:
                index = max((min((index, np.size(self.vals, axis=axis) - 1)), 0))
                if mode is None or mode == 'insert':
                    self.vals = np.insert(self.vals, index,
                                          new_vals, axis)
                elif mode == 'delete':
                    self.vals = np.delete(self.vals, index, axis)
                else:
                    print('Mode {} not understood'.format(mode))

    def generate_zmesh(self, min_z, max_z, NZ):

        self.dz = utils.generate_zmesh(min_z, max_z, NZ)[0]
        if not self.is_half_space():
            self.vals = np.zeros((self.nx, self.ny, self.nz)) + self.background_resistivity
            print('Mesh generation not setup for non-half-spaces. Converting to half space...')
        # self.dz = zmesh

    def generate_mesh(self, site_locs, regular=True, min_x=None, min_y=None,
                      max_x=None, max_y=None, num_pads=None, pad_mult=None):
        self._dx = list(utils.generate_lateral_mesh(site_locs[:, 0],
                                                   min_x=min_x, max_x=max_x, regular=regular)[0])
        self._dy = list(utils.generate_lateral_mesh(site_locs[:, 1],
                                                   min_x=min_y, max_x=max_y, regular=regular)[0])
        if not self.is_half_space():
            print('Mesh generation not setup for non-half-spaces. Converting to half space...')
            self.vals = np.zeros((self.nx, self.ny, self.nz)) + self.background_resistivity
        # self.dy = ymesh
        # self.dx = xmesh

    def check_new_mesh(self, attr, index, value):
        mesh = getattr(self, attr)
        if index > 0:
            if value <= mesh[index - 1]:
                print('Unable to add mesh: Mesh locations must be strictly increasing')
                return False
        else:
            if value >= mesh[index]:
                print('Unable to add mesh: Mesh locations must be strictly increasing')
                return False
        return True

    # def insert_mesh(self, dim, index, value):
    #     ACCEPTED_DIMENSIONS = {'dx': 0, 'dy': 1, 'dz': 2,
    #                            'xcs': 0, 'ycs': 1, 'zcs': 2}
    #     if dim.lower() not in ACCEPTED_DIMENSIONS.keys():
    #         print('Dimension not recognized.')
    #     else:
    #         getattr(self,)
    def dx_delete(self, index):
        del self._dx[index]
        self._xCS = list(np.diff(self._dx))
        self.update_vals(axis=0, index=index, mode='delete')

    def dy_delete(self, index):
        del self._dy[index]
        self._yCS = list(np.diff(self._dy))
        self.update_vals(axis=1, index=index, mode='delete')

    def dz_delete(self, index):
        del self._dz[index]
        self._zCS = list(np.diff(self._dz))
        self.update_vals(axis=2, index=index, mode='delete')

    def dx_insert(self, index, value):
        if self.check_new_mesh('dx', index, value):
            mod_idx = max((min((index, self.nx - 1)), 0))
            new_vals = self.vals[max((mod_idx - 1, 0)), :, :]
            self._dx = self._dx[:index] + [value] + self._dx[index:]
            self._xCS = list(np.diff(self._dx))
            self.update_vals(axis=0, index=mod_idx, new_vals=new_vals)

    def dy_insert(self, index, value):
        if self.check_new_mesh('dy', index, value):
            mod_idx = max((min((index, self.ny - 1)), 0))
            new_vals = self.vals[:, max((mod_idx - 1, 0)), :]
            self._dy = self._dy[:index] + [value] + self._dy[index:]
            self._yCS = list(np.diff(self._dy))
            self.update_vals(axis=1, index=mod_idx, new_vals=new_vals)

    def dz_insert(self, index, value):
        if self.check_new_mesh('dz', index, value):
            mod_idx = max((min((index, self.nz - 1)), 0))
            new_vals = self.vals[:, :, max((mod_idx - 1, 0))]
            self._dz = self._dz[:index] + [value] + self._dz[index:]
            self._zCS = list(np.diff(self._dz))
            self.update_vals(axis=2, index=mod_idx, new_vals=new_vals)
# start Properties

    @property
    def nx(self):
        return len(self._xCS)

    @property
    def ny(self):
        return len(self._yCS)

    @property
    def nz(self):
        return len(self._zCS)

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, vals):
        if len(set(vals)) != len(vals):
            print('Mesh lines not unique. Ignoring request.')
            return
        self._dx = vals
        self._xCS = list(np.diff(self._dx))
        self.update_vals()

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, vals):
        if len(set(vals)) != len(vals):
            print('Mesh lines not unique. Ignoring request.')
            return
        self._dy = vals
        self._yCS = list(np.diff(self._dy))
        self.update_vals()

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, vals):
        if len(set(vals)) != len(vals):
            print('Mesh lines not unique. Ignoring request.')
            return
        self._dz = vals
        self._zCS = list(np.diff(self._dz))
        self.update_vals()

    @property
    def xCS(self):
        return self._xCS

    @xCS.setter
    def xCS(self, vals):
        self._xCS = vals
        self._dx = list(np.cumsum([0, *vals]) - np.sum(vals) / 2)
        self.update_vals()

    @property
    def yCS(self):
        return self._yCS

    @yCS.setter
    def yCS(self, vals):
        self._yCS = vals
        self._dy = list(np.cumsum([0, *vals]) - np.sum(vals) / 2)
        self.update_vals()

    @property
    def zCS(self):
        return self._zCS

    @zCS.setter
    def zCS(self, vals):
        self._zCS = vals
        self._dz = list(np.cumsum([0, *vals]))
        self.update_vals()

    def write(self, outfile):
        WS_io.write_model(self, outfile)


class Response(Data):
    """Summary
    """

    def write(self, outfile):
        WS_io.write_response(self, outfile)


class Site(object):
    """Contains data for a given site.

    Attributes:
        ACCEPTED_COMPONENTS (TYPE): Description
        components (TYPE): Description
        data (TYPE): Description
        errmap (dict): Description
        errors (TYPE): Description
        locations (TYPE): Description
        name (TYPE): Description
        pColours (list): Description
        periods (TYPE): Description
        used_error (dict): Description
        name - The name of the site.
        data - The data recorded at the site. The data attribute is a dictionary whose keys
               are the data components, and whose values is a numpy array of points
               corresponding to the periods in self.periods.
        errors - The error information for the data. the error attribute has the
                 same structure as the data attribute.
        errmap - The error map for the data at this site. If the data is raw data,
                 the map is set to all ones.
        periods - The periods at which data is available.
        components - The data components available for this site.
    """
    ACCEPTED_COMPONENTS = ('ZXXR', 'ZXXI',
                           'ZXYR', 'ZXYI',
                           'ZYYR', 'ZYYI',
                           'ZYXR', 'ZYXI',
                           'TZXR', 'TZXI',
                           'TZYR', 'TZYI')
    HIGHFREQ_FLAG = -9
    DIAG_FLAG = -99
    OUTLIER_FLAG = -999
    NO_PERIOD_FLAG = -9999
    NO_COMP_FLAG = -99999

    def __init__(self, data={}, name='', periods=None, locations={},
                 errors={}, errmap=None, azimuth=None, flags=None,
                 errfloorZ=None, errfloorT=None, solve_static=0):
        """Initialize a Site object.
        Data must a dictionary where the keys are acceptable tensor components,
        and the length matches the number of periods.

        Args:
            data (dict, optional): Description
            name (str, optional): Description
            periods (None, optional): Description
            locations (list, optional): Description
            errors (dict, optional): Description
            errmap (dict, optional): Description
        """
        self.name = name
        self.data = data
        self.periods = utils.truncate(periods)
        self.locations = locations
        self.errors = errors
        self.components = [key for key in data.keys()]
        self.orig_azimuth = azimuth
        self.azimuth = azimuth
        self.solve_static = solve_static
        self.static_multiplier = 1
        self.error_floors = {'Z': 0.075,
                             'T': 0.15,
                             'Rho': 0.05,
                             'Phs': 0.05}
        self.phase_tensors = []
        if errfloorZ is None:
            self.errfloorZ = 5
        else:
            self.errfloorZ = errfloorZ
        if errfloorT is None:
            self.errfloorT = 15
        else:
            self.errfloorT = errfloorT
        if errmap:
            self.errmap = errmap
        else:
            self.errmap = self.generate_errmap(mult=1)
        if flags:
            self.flags = flags
        else:
            self.flags = self.generate_errmap(mult=1)
        if not self.errors or utils.is_all_empties(self.errors):
            self.errors = self.generate_errmap(mult=1)
        if not self.errmap or utils.is_all_empties(self.errmap):
            self.errmap = self.generate_errmap(mult=1)
        self.calculate_phase_tensors()

            # self.errmap = {}
            # self.flags = {}
            # for comp in self.data.keys():
            #     self.errmap.update({comp:
            #                         np.ones(np.shape(self.data[comp]))})
        try:
            self.used_error = {component: self.errors[component] for component in self.components}
        except KeyError:
            print('Error information not available.')
            self.used_error = {}
        # Add dummy data to missing components
        self.apply_error_floor()
        # for comp, val in self.errors.items():
        #     self.used_error.update({comp: val * self.errmap[comp]})

        # Rotate all sites to 0 degrees to start
        # self.rotate_data(azi=0)

    @property
    def errorfloorZ(self):
        return self.error_floors['Z']

    @errorfloorZ.setter
    def errorfloorZ(self, val):
        self.error_floors['Z'] = val

    @property
    def errorfloorT(self):
        return self.error_floors['T']

    @errorfloorT.setter
    def errorfloorT(self, val):
        self.error_floors['T'] = val

    def calculate_error_floor(self, error_floor=None, components=None):
        if error_floor is None:
            error_floor = self.error_floors
        if not components:
            components = self.components
        elif not isinstance(components, list):
            components = [components]
        error_floors = self.error_floors
        if error_floors is not None:
            try:
                for key in error_floors.keys():
                    error_floors[key] = error_floor[key]
            except TypeError:
                for key in error_floors.keys():
                    error_floors[key] = error_floor
        returned_errors = {component: [] for component in components}
        for component in components:
            if 'log10' in component:
                new_errors = np.log10(1 + error_floors['Rho']) * np.ones(self.data[component].shape)
            elif 'phs' in component.lower():
                new_errors = error_floors['Phs'] * 100 * np.ones(self.data[component].shape)
            elif component.startswith('Z'):
                new_errors = np.abs(self.data[component] * error_floors['Z'])
            elif component.startswith('T'):
                new_errors = np.abs(self.data[component] * error_floors['T'])
            elif 'rho' in component.lower():
                new_errors = np.abs(self.data[component] * error_floors['Rho'])
            new_errors[new_errors == 0] = np.median(new_errors)
            returned_errors[component] = utils.truncate(new_errors)
        return returned_errors

    def reset_errors(self, error_floor=None, components=None):
        if not components:
            components = self.components
        new_errors = self.calculate_error_floor(error_floor=error_floor,
                                                components=components)
        for component in components:
            self.errors[component] = new_errors[component]
            self.used_error[component] = new_errors[component]

    @utils.enforce_input(noise_level=float, components=list)
    def add_noise(self, noise_level=5, components=None):
        if noise_level > 1:
            noise_level /= 100
        if not components:
            components = self.components
        for component in components:
            size = self.data[component].shape
            if 'phs' in component.lower():
                noise = np.random.normal(loc=noise_level * 100,
                                         scale=noise_level * 100, size=size)
            elif 'log10' in component.lower():
                noise = np.random.normal(loc=np.log10(1 + noise_level),
                                         scale=np.log10(1 + noise_level), size=size)
            else:
                noise = []
                for val in self.data[component]:
                    noise.append(np.random.normal(loc=abs(val) * noise_level,
                                                  scale=abs(val) * noise_level, size=1))
                noise = np.array(noise)
            noise *= np.random.choice((1, -1))
            self.data[component] += np.squeeze(noise)
            self.data[component] = utils.truncate(self.data[component])

    def apply_error_floor(self, error_floors=None, errfloorZ=None, errfloorT=None):
        if errfloorZ:
            print('Usage of errfloorZ is depreciated and will be removed in the future.\n')
            print('Please either set the error_floor attribute, or pass a dictionary argument ' +
                  ' as error_floor')
            self.error_floors['Z'] = errfloorZ
        if errfloorT:
            print('Usage of errfloorT is depreciated and will be removed in the future.\n')
            print('Please either set the error_floor attribute, or pass a dictionary argumuent ' +
                  ' as error_floor')
            self.error_floors['T'] = errfloorT
        if error_floors:
            try:
                for key in error_floors.keys():
                    self.error_floors[key] = error_floors[key]
            except TypeError:
                for key in self.error_floors.keys():
                    self.error_floors[key] = error_floors
        floor_errors = self.calculate_error_floor(self.error_floors)
        for component in self.components:
            new_errors = np.maximum.reduce([floor_errors[component],
                                            self.errors[component] * self.errmap[component]])
            self.used_error[component] = new_errors

        # if errfloorZ is None:
        #     errfloorZ = self.errfloorZ
        # else:
        #     self.errfloorZ = errfloorZ
        # if errfloorT is None:
        #     errfloorT = self.errfloorT
        # else:
        #     self.errfloorT = errfloorT
        # comps = set([comp[:3] for comp in self.components])
        # if 'ZXY' in comps:
        #     ZXY = self.data['ZXYR'] + 1j * self.data['ZXYI']
        #     ZYX = self.data['ZYXR'] + 1j * self.data['ZYXI']
        #     erabsz = np.sqrt(np.absolute(ZXY * ZYX)) * errfloorZ / 100
        #     mapped_err = {comp: self.errors[comp] * self.errmap[comp] for
        #                   comp in self.errors if comp[0] == 'Z'}
        #     for comp, err in mapped_err.items():
        #         self.used_error[comp] = np.array([max(fl, mapped)
        #                                           for (fl, mapped) in zip(erabsz, err)])
        # if 'TZX' in comps:
        #     TZX = self.data['TZXR'] + 1j * self.data['TZXI']
        #     TZY = self.data['TZYR'] + 1j * self.data['TZYI']
        #     ERT = np.sqrt(np.absolute(TZX * TZX) + np.absolute(TZY * TZY))
        #     erabst = ERT * errfloorT / 100
        #     mapped_err = {comp: self.errors[comp] * self.errmap[comp] for
        #                   comp in self.errors if comp[0] == 'T'}
        #     for comp, err in mapped_err.items():
        #         self.used_error[comp] = np.array([max(fl, mapped) for (fl, mapped) in zip(erabst, err)])

    def rotate_data(self, azi=0):
        # This function is needlessly long, but I want to make sure it does it's job right...
        def rotz(zxx, zxy, zyx, zyy, theta):
            theta = 2 * np.deg2rad(theta)  # Make use of double angle identities
            return ((zxx + zyy) + (zxx - zyy) * np.cos(theta) + (zxy + zyx) * np.sin(theta)) / 2
        if set(['ZXXR', 'ZXXI', 'ZXYR', 'ZXYI',
                'ZYXR', 'ZYXI', 'ZYYR', 'ZYYI']).issubset(set(self.components)):
            if azi < 0:
                azi += 360
            theta = azi - self.azimuth
            if theta < 0:
                theta += 360
            if theta == 0:
                if DEBUG:
                    print('{}: No rotation required.'.format(self.name))
                return
            else:
                if DEBUG:
                    print('Rotating site {} by {} degrees'.format(self.name, theta))
            # print('Azi: {}, Theta:{}, Self.Azi: {}'.format(azi, theta, self.azimuth))
            zxxR = {'data': [], 'errors': []}
            zxyR = {'data': [], 'errors': []}
            zyxR = {'data': [], 'errors': []}
            zyyR = {'data': [], 'errors': []}
            for dtype in ['data', 'errors']:
                zxx = getattr(self, dtype)['ZXXR'] + 1j * getattr(self, dtype)['ZXXI']
                zxy = getattr(self, dtype)['ZXYR'] + 1j * getattr(self, dtype)['ZXYI']
                zyx = getattr(self, dtype)['ZYXR'] + 1j * getattr(self, dtype)['ZYXI']
                zyy = getattr(self, dtype)['ZYYR'] + 1j * getattr(self, dtype)['ZYYI']
                zxxR[dtype] = rotz(zxx, zxy, zyx, zyy, theta)
                zxyR[dtype] = rotz(zxy, -zxx, zyy, -zyx, theta)
                zyxR[dtype] = rotz(zyx, zyy, -zxx, -zxy, theta)
                zyyR[dtype] = rotz(zyy, -zyx, -zxy, zxx, theta)
            self.data['ZXXR'], self.data['ZXXI'] = [np.real(zxxR['data']), np.imag(zxxR['data'])]
            self.data['ZXYR'], self.data['ZXYI'] = [np.real(zxyR['data']), np.imag(zxyR['data'])]
            self.data['ZYXR'], self.data['ZYXI'] = [np.real(zyxR['data']), np.imag(zyxR['data'])]
            self.data['ZYYR'], self.data['ZYYI'] = [np.real(zyyR['data']), np.imag(zyyR['data'])]
            self.errors['ZXXR'], self.errors['ZXXI'] = [np.real(zxxR['errors']),
                                                        np.imag(zxxR['errors'])]
            self.errors['ZXYR'], self.errors['ZXYI'] = [np.real(zxyR['errors']),
                                                        np.imag(zxyR['errors'])]
            self.errors['ZYXR'], self.errors['ZYXI'] = [np.real(zyxR['errors']),
                                                        np.imag(zyxR['errors'])]
            self.errors['ZYYR'], self.errors['ZYYI'] = [np.real(zyyR['errors']),
                                                        np.imag(zyyR['errors'])]
            if 'TZXR' in self.components:
                theta_rad = np.deg2rad(theta)  # This is what was missing... Obviously it was something dumb.
                self.data['TZXR'], self.data['TZYR'] = (self.data['TZXR'] * np.cos(theta_rad) -
                                                        self.data['TZYR'] * np.sin(theta_rad),
                                                        self.data['TZXR'] * np.sin(theta_rad) +
                                                        self.data['TZYR'] * np.cos(theta_rad))
                self.data['TZXI'], self.data['TZYI'] = (self.data['TZXI'] * np.cos(theta_rad) -
                                                        self.data['TZYI'] * np.sin(theta_rad),
                                                        self.data['TZXI'] * np.sin(theta_rad) +
                                                        self.data['TZYI'] * np.cos(theta_rad))
                # tX = self.data['TZXR'] * np.cos(theta) - self.data['TZYR'] * np.sin(theta)
                # tY = self.data['TZXR'] * np.sin(theta) + self.data['TZYR'] * np.cos(theta)
                # self.data['TZXR'] = tX
                # self.data['TZYR'] = tY
                # tX = self.data['TZXI'] * np.cos(theta) - self.data['TZYI'] * np.sin(theta)
                # tY = self.data['TZXI'] * np.sin(theta) + self.data['TZYI'] * np.cos(theta)
                # self.data['TZXI'] = tX
                # self.data['TZYI'] = tY

                self.errors['TZXR'], self.errors['TZYR'] = (self.errors['TZXR'] * np.cos(theta) -
                                                            self.errors['TZYR'] * np.sin(theta),
                                                            self.errors['TZXR'] * np.sin(theta) +
                                                            self.errors['TZYR'] * np.cos(theta))
                self.errors['TZXI'], self.errors['TZYI'] = (self.errors['TZXI'] * np.cos(theta) -
                                                            self.errors['TZYI'] * np.sin(theta),
                                                            self.errors['TZXI'] * np.sin(theta) +
                                                            self.errors['TZYI'] * np.cos(theta))
            self.azimuth = azi
        else:
            print('Cannot rotate data. Required components not available.')

    def add_periods(self, site):
        """
        Adds data from missing periods in self from site. If only certain periods are needed, use RawData.get_data
        to generate a Site object with the desired periods.

        Args:
            data (TYPE): Description
            new_periods (TYPE): Description
            errors (None, optional): Description
            errmap (None, optional): Description
        """
        shift = 0
        data = site.data
        new_periods = site.periods
        errors = site.errors
        errmap = site.errmap
        if errors is None:
            errors = 0.05
        if errmap is None:
            errmap = 1
        new_periods = utils.to_list(new_periods)
        current_periods = list(self.periods)
        # if len(errors) <= 1:
        #     errors = Data.generate_errors(data, errors)
        # if len(errmap) <= 1:
        #     errmap = Data.generate_errmap(data, errmap)
        new_data = {comp: utils.np2list(self.data[comp]) for comp in self.components}
        new_errors = {comp: utils.np2list(self.errors[comp]) for comp in self.components}
        new_errmap = {comp: utils.np2list(self.errmap[comp]) for comp in self.components}
        for comp in self.components:
            for ii, period in enumerate(new_periods):
                ind = [jj for jj, val in enumerate(current_periods) if val > period]
                try:
                    ind = ind[0]
                except IndexError:
                    ind = len(current_periods)
                new_data[comp].insert(ind - shift, data[comp][ii])
                new_errors[comp].insert(ind - shift, errors[comp][ii])
                new_errmap[comp].insert(ind - shift, errmap[comp][ii])
                if period not in current_periods:
                    current_periods.insert(ind, period)
                    shift = 1
        self.data = {comp: utils.list2np(new_data[comp]) for comp in self.components}
        self.errors = {comp: utils.list2np(new_errors[comp]) for comp in self.components}
        self.errmap = {comp: utils.list2np(new_errmap[comp]) for comp in self.components}
        self.periods = utils.list2np(utils.truncate(current_periods))
        self.apply_error_floor()

    @utils.enforce_input(periods=list)
    def remove_periods(self, periods):
        """Summary

        Args:
            periods (TYPE): Description

        Returns:
            TYPE: Description
        """
        # periods = utils.to_list(periods)
        for period in periods:
            # if any(np.isclose(period, self.periods)):
            # ind = [ii for ii, val in enumerate(self.periods) if val >= period][0]
            ind = np.where(self.periods == period)[0]
            # if ind is None
            # self.periods.delete(ind)
            self.periods = np.delete(self.periods, ind)
            for comp in self.data.keys():
                self.data[comp] = np.delete(self.data[comp], ind)
                self.errors[comp] = np.delete(self.errors[comp], ind)
                self.errmap[comp] = np.delete(self.errmap[comp], ind)
                self.used_error[comp] = np.delete(self.used_error[comp], ind)

    def add_component(self, data, errors=0.05, errmap=1):
        """
        Adds component data to the site. Data must be in the standard dictionary format
        of {comp1: data, comp2: data, ...}
        Data must be given, but error and error maps default to 5% of data, and ones, respectively.

        Args:
            data (TYPE): Description
            errors (float, optional): Description
            errmap (int, optional): Description
        """
        if not isinstance(errors, dict):
            errors = Site.generate_errors(data, errors)
        if not isinstance(errmap, dict):
            errmap = self.generate_errmap(errmap)

        for comp in data.keys():
            self.data.update({comp: data[comp]})
            self.errors.update({comp: errors[comp]})
            self.errmap.update({comp: errmap[comp]})
            self.components.append(comp)

    @staticmethod
    def generate_errors(data, mult=0.05):
        """Summary

        Args:
            data (TYPE): Description
            mult (float, optional): Description

        Returns:
            TYPE: Description
        """
        return {comp: val * mult for comp, val in data.items()}

    def generate_errmap(self, mult=1):
        """
        Returns an error map dictionary with the same structure as the input data.
        If a single value is entered for 'mult', all error map values are set to that.
        Alternatively, a dictionary of {comp: val} may be input, and componenets will
        have their error maps set accordingly.

        Args:
            data (TYPE): Description
            mult (int, optional): Description
        """
        if not isinstance(mult, dict):
            return {comp: mult * np.ones(np.shape(val)) for comp, val in self.data.items()}
        else:
            return {comp: mult[comp] * np.ones(np.shape(val)) for comp, val in self.data.items()}

    @utils.enforce_input(periods=list, comps=list, mult=int, multiplicative=bool)
    def change_errmap(self, periods, comps, mult, multiplicative=True):
        for comp in comps:
            for period in periods:
                ind = np.argmin(abs(self.periods - period))
                if multiplicative:
                    self.errmap[comp][ind] = mult * self.errmap[comp][ind]
                else:
                    self.errmap[comp][ind] = mult
                self.used_error[comp][ind] = self.errors[comp][ind] * self.errmap[comp][ind]
        self.apply_error_floor()

    @utils.enforce_input(components=list)
    def remove_components(self, components):
        """Summary

        Args:
            component (TYPE): Description

        Returns:
            TYPE: Description
        """
        # component = utils.to_list(component)
        for comp in components:
            try:
                del self.data[comp]
                del self.errors[comp]
                del self.errmap[comp]
                del self.flags[comp]
                self.components.remove(comp)
            except AttributeError:
                print('Component {} doesn\'t exist'.format(comp))

    def loosely_equal(self, site):
        """
        Compares two sites to check for equality. For this method, equality just means that
        the sites have the same components and periods

        Args:
            site (TYPE): Description
        """
        if set(self.components) == set(site.components) and \
                all(self.periods == site.periods):
            return True
        else:
            return False

    def strictly_equal(self, site):
        """Summary

        Args:
            site (TYPE): Description

        Returns:
            TYPE: Description
        """
        # if self.loosely_equal(site):
        # First check for loose equality
        # Equality = namedtuple('Equality', ['loosely', 'data',
        #                                    'errors', 'errmap', 'locations',
        #                                    'names'])
        retval = {'loosely': True, 'data': True,
                  'errors': True, 'errmap': True,
                  'locations': True, 'names': True}
        retval['loosely'] = self.loosely_equal(site)
        for comp in self.components:
            if not np.all(np.isclose(self.data[comp], site.data[comp])):
                retval['data'] = False
            if not np.all(np.isclose(self.errors[comp], site.errors[comp])):
                retval['errors'] = False
            if not np.all(self.errmap[comp] == site.errmap[comp]):
                retval['errmap'] = False
            if not (self.locations['X'] == site.locations['X'] and
                    self.locations['Y'] == site.locations['Y']):
                retval['locations'] = False
            if self.name != site.name:
                retval['names'] = False
        return retval

    @utils.enforce_input(periods=list, components=list, lTol=float, hTol=float)
    def get_data(self, periods=None, components=None, lTol=0.02, hTol=0.10):
        if periods is None:
            periods = self.periods
        if components is None:
            components = self.components
        if not set(components).issubset(set(Site.ACCEPTED_COMPONENTS)):
            invalid_component = set(Site.ACCEPTED_COMPONENTS) - set(components)
            print('{} is not a valid component. Ignoring...'.format(invalid_component))
            components = [comp for comp in components if comp not in invalid_component]
        if lTol is None:
            lTol = 0.02
        if hTol is None:
            hTol = 0.10
        name = self.name
        locations = self.locations
        data = {comp: [] for comp in components}
        errors = {comp: [] for comp in components}
        errmap = {comp: [] for comp in components}
        flags = {comp: [] for comp in components}
        azimuth = self.azimuth
        periods = sorted(periods)
        for comp in components:
            d = []
            e = []
            em = []
            f = []
            for p in periods:
                ind = np.argmin(abs(self.periods - p))
                if (p <= 1 and utils.percdiff(self.periods[ind], p) > lTol) or \
                   (p > 1 and utils.percdiff(self.periods[ind], p) > hTol):
                    mult = self.NO_PERIOD_FLAG  # Flag for missing period
                else:
                    mult = 1
                try:
                    d.append(self.data[comp][ind])
                    e.append(self.errors[comp][ind])
                    em.append(self.errmap[comp][ind] * mult)
                    f.append(mult)
                except KeyError:
                    d.append(np.float(self.NO_COMP_FLAG))
                    e.append(np.float(self.NO_COMP_FLAG))
                    em.append(np.float(self.NO_COMP_FLAG))
                    f.append(np.float(self.NO_COMP_FLAG))
            # Check for dummy TF data
            if comp[0] == 'T':
                if all(abs(x) < 0.001 for x in d):
                    em = [self.NO_COMP_FLAG for x in em]
            data.update({comp: np.array(d)})
            errors.update({comp: np.array(e)})
            errmap.update({comp: np.array(em)})
            flags.update({comp: np.array(f)})
        periods = np.array(periods)
        return Site(data=data, errors=errors, errmap=errmap, periods=periods,
                    locations=locations, azimuth=azimuth, name=name, flags=flags)

    def detect_outliers(self, outlier_map=10):
        nper = len(self.periods)
        for comp in self.components:
            for idx, per in enumerate(self.periods):
                datum = self.data[comp][idx]
                expected = 0
                jj = 1
                for jj in range(max(1, idx - 2), min(nper, idx + 2)):
                    expected += self.data[comp][jj]
                expected /= jj
                tol = abs(2 * expected)
                diff = datum - expected
                if abs(diff) > tol:
                    self.errmap[comp][idx] *= outlier_map
                    self.flags[comp][idx] = self.OUTLIER_FLAG

    def calculate_phase_tensors(self):
        self.phase_tensors = []
        for ii, period in enumerate(self.periods):
            Z = {impedance: self.data[impedance][ii] for impedance in self.data.keys()}
            self.phase_tensors.append(PhaseTensor(period=period, Z=Z))


class RawData(object):
    """Summary

    Attributes:
        datpath (TYPE): Description
        listfile (TYPE): Description
        master_periods (TYPE): Description
        narrow_periods (TYPE): Description
        RAW_COMPONENTS (TYPE): Description
        site_names (TYPE): Description
        sites (TYPE): Description
    """
    RAW_COMPONENTS = ('ZXX', 'ZXY',
                      'ZYX', 'ZYY',
                      'TZX', 'TZY')

    def __init__(self, listfile='', datpath=''):
        """Summary

        Args:
            listfile (TYPE): Description
            datpath (str, optional): Description
        """
        if not datpath and listfile:
            datpath = os.path.dirname(listfile)
        self.datpath = datpath
        self.listfile = listfile
        if listfile:
            self.__read__(listfile, datpath)
            self.initialized = True
        else:
            self.sites = {}
            self.datpath = ''
            self.site_names = []
            self.locations = []
            self.initialized = False
        self.low_tol = 0.02
        self.high_tol = 0.1
        self.count_tol = 0.5
        self.master_periods = self.master_period_list()
        self.narrow_periods = self.narrow_period_list()
        self.azimuth = 0
        for site_name in self.site_names:
            self.sites[site_name].rotate_data(azi=0)

    @property
    def origin(self):
        min_x, max_x = (min(self.locations[:, 1]), max(self.locations[:, 1]))
        min_y, max_y = (min(self.locations[:, 0]), max(self.locations[:, 0]))
        origin = (min_x + (max_x - min_x) / 2,
                  min_y + (max_y - min_y) / 2)
        return origin

    def __read__(self, listfile, datpath=''):
        """Summary

        Args:
            listfile (TYPE): Description
            datpath (str, optional): Description

        Returns:
            TYPE: Description
        """
        self.site_names = WS_io.read_sites(listfile)
        all_data = WS_io.read_raw_data(self.site_names, datpath)
        self.sites = {}
        for site_name, site in all_data.items():
            self.sites.update({site_name:
                               Site(name=site_name,
                                    data=site['data'],
                                    errors=site['errors'],
                                    errmap=site.get('errmap', None),
                                    periods=site['periods'],
                                    locations=site['locations'],
                                    azimuth=site['azimuth'],
                                    flags=None
                                    )})
        dummy_sites = self.check_dummy_data(threshold=0.001)
        if dummy_sites:
            self.remove_components(sites=dummy_sites,
                                   components=['TZXR', 'TZXI', 'TZYR', 'TZYI'])
        #  Check this. It looks like more periods are being removed than should be?
        #  Take out the call to 'remove_periods' and manually check what it wants to take out.
        dummy_periods = self.check_dummy_periods()
        if dummy_periods:
            self.remove_periods(site_dict=dummy_periods)
        self.locations = Data.get_locs(self)
        self.datpath = datpath
        self.listfile = listfile

    def remove_periods(self, site_dict):
        for site, periods in site_dict.items():
            self.sites[site].remove_periods(periods=periods)

    def remove_components(self, sites=None, components=None):
        for site in sites:
            self.sites[site].remove_components(components=components)

    def check_dummy_periods(self, threshold=1e-10):
        sites = {}
        for site in self.sites.values():
            periods = []
            for ii, p in enumerate(site.periods):
                vals = [site.data[comp][ii] for comp in site.components if comp[0] == 'Z']
                if all(abs(abs(np.array(vals)) - abs(vals[0])) < threshold):
                    periods.append(p)
            if periods:
                sites.update({site.name: periods})
        return sites

    def check_dummy_data(self, threshold=0.001):
        # for comp in ('TZXR', 'TZXI', 'TZYR', 'TZYI'):
        # Only need to check one, right?
        sites = []
        for site in self.sites.values():
            if 'TZXR' in site.components:
                if np.all(abs(site.data['TZXR']) < threshold):
                    sites.append(site.name)
        return sites

    def get_locs(self, sites=None, azi=0, mode=None):
        if not mode:
            mode = 'utm'
        if mode.lower() == 'utm' or mode.lower() == 'centered':
            X = 'X'
            Y = 'Y'
        elif mode.lower() == 'latlong':
            X = 'Lat'
            Y = 'Long'
        if sites is None:
            sites = self.site_names
        locs = np.array([[self.sites[name].locations[X],
                          self.sites[name].locations[Y]]
                         for name in sites])
        if azi != 0:
            locs = utils.rotate_locs(locs, azi)
        if mode.lower() == 'centered':
            locs = utils.center_locs(locs)[0]
        return locs

    def master_period_list(self):
        """Summary

        Returns:
            TYPE: Description
        """
        periods = []
        for site in self.sites.values():
            periods.append(list(site.periods))
        periods = [utils.truncate(p) for sublist in periods for p in sublist]
        lp = len(self.sites)
        for period in set(periods):
            periods.count(period)
        return {utils.truncate(period):
                utils.truncate(periods.count(period) / lp) for period in set(periods)}

    def narrow_period_list(self, periods=None, count_tol=None, low_tol=None, high_tol=None):
        """Summary

        Args:
            periods (None, optional): The list of periods to narrow down.
            If None, master_periods is used.
            count_tol (float, optional): Minimum fraction of sites that must contain the period.
            low_tol (float, optional): Wiggle room on periods less than 1 second.
            high_tol (float, optional): Wiggle room on periods greater than 1 second.

        Returns:
            TYPE: Returns a narrowed periods list
        """
        self.set_tols(hTol=high_tol, lTol=low_tol, cTol=count_tol)
        if periods is None:
            periods = [p for p, v in self.master_periods.items() if v >= self.count_tol]
        counter = {period: 0 for period in periods}
        for site in self.sites.values():
            for p in periods:
                if p < 1:
                    mult = self.low_tol
                else:
                    mult = self.high_tol
                if any((site.periods >= p - p * mult) * (site.periods <= p + p * mult)):
                    counter[p] += 1
        return {utils.truncate(p):
                utils.truncate(val / len(self.sites)) for p, val in counter.items()
                if val / len(self.sites) >= self.count_tol}

    @utils.enforce_input(sites=list, periods=list, components=list, lTol=float, hTol=float)
    def get_data(self, sites=None, periods=None, components=None,
                 lTol=None, hTol=None):
        self.set_tols(hTol=hTol, lTol=lTol, cTol=None)
        if sites is None:
            sites = self.sites
        # elif isinstance(sites, str):
        else:
            # sites = {sites: self.sites[sites]}
            sites = {site_name: self.sites[site_name] for site_name in sites}
        if periods is None:
            periods = list(self.narrow_periods.keys())
        if components is None:
            components = []
            for comp in self.RAW_COMPONENTS:
                components.append(''.join([comp, 'R']))
                components.append(''.join([comp, 'I']))
        if len(periods) > 16:
            minp = np.log10(min(periods))
            maxp = np.log10(max(periods))
            logp = np.logspace(minp, maxp, 12)
            periods = utils.closest_periods(wanted_p=logp, available_p=periods)
        ret_sites = {}
        for site in sites.values():
            ret_sites.update({site.name: site.get_data(periods=periods, components=components,
                                                       lTol=self.low_tol, hTol=self.high_tol)})
        return ret_sites

    def rotate_sites(self, azi=0):
        if azi != self.azimuth:
            if azi < 0:
                azi = 360 + azi
            theta = self.azimuth - azi
            self.azimuth = azi
            locs = self.locations
            self.locations = utils.rotate_locs(locs, theta)
            for site_name in self.site_names:
                self.sites[site_name].rotate_data(azi)

    def set_tols(self, hTol, lTol, cTol):
        if hTol is not None:
            self.high_tol = hTol
        if lTol is not None:
            self.low_tol = lTol
        if hTol is not None:
            self.count_tol = cTol

    def remove_sites(self, sites):
        """Summary

        Args:
            site (TYPE): Description

        Returns:
            TYPE: Description
        """
        self = Data.remove_sites(self, sites=sites)
        self.master_periods = self.master_period_list()
        self.narrow_periods = self.narrow_period_list()

    def add_site(self, site):
        self.sites.update({site.name: site})
        self.site_names.append(site.name)
        self.master_periods = self.master_period_list()
        self.narrow_periods = self.narrow_period_list()

    def check_compromised_data(self, threshold=0.75):
        return Data.check_compromised_data(self, threshold=threshold)

    def to_vtk(self, outfile, UTM, origin=None, sea_level=0):
        if not origin:
            origin = self.origin
        Data.to_vtk(self, outfile=outfile, origin=origin, UTM=UTM, sea_level=sea_level)


class PhaseTensor(object):
    def __init__(self, period, Z):
        self.X, self.Y = np.zeros((2, 2)), np.zeros((2, 2))
        self.period = period
        self.Z = Z
        self.det_phi = 0
        self.skew_phi = 0
        self.phi_1 = 0
        self.phi_2 = 0
        self.phi_max = 0
        self.phi_min = 0
        self.alpha = 0
        self.beta = 0
        self.Lambda = 0
        self.azimuth = 0
        self.form_tensors(Z)
        self.calculate_phase_tensor()
        self.calculate_phase_parameters()

    def form_tensors(self, Z):
        self.X = np.array(((Z['ZXXR'], Z['ZXYR']),
                           (Z['ZYXR'], Z['ZYYR'])))
        self.Y = np.array(((Z['ZXXI'], Z['ZXYI']),
                           (Z['ZYXI'], Z['ZYYI'])))

    def calculate_phase_tensor(self):
        self.phi = np.matmul(np.linalg.inv(self.X), self.Y)

    def calculate_phase_parameters(self):
        det_phi = (np.linalg.det(self.phi))
        skew_phi = (self.phi[0, 1] - self.phi[1, 0])
        phi_1 = np.trace(self.phi) / 2
        phi_2 = np.sqrt(det_phi)
        phi_3 = skew_phi / 2
        phi_max = np.sqrt(phi_1 ** 2 + phi_3 ** 2) + np.sqrt(phi_1 ** 2 + phi_3 ** 2 - det_phi)
        phi_min = np.sqrt(phi_1 ** 2 + phi_3 ** 2) - np.sqrt(phi_1 ** 2 + phi_3 ** 2 + det_phi)
        alpha = 0.5 * np.arctan((self.phi[0, 1] + self.phi[1, 0]) / (self.phi[0, 0] - self.phi[1, 1]))
        Lambda = (phi_max - phi_min) / (phi_max + phi_min)
        beta = 0.5 * np.arctan(phi_3 / phi_1)
        azimuth = 0.5 * np.pi - (alpha - beta)
        self.det_phi = det_phi
        self.skew_phi = skew_phi
        self.phi_1 = phi_1
        self.phi_2 = phi_2
        self.phi_3 = phi_3
        self.phi_max = phi_max
        self.phi_min = phi_min
        self.alpha = alpha
        self.Lambda = Lambda
        self.beta = beta
        self.azimuth = azimuth
