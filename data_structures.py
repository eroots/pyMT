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

    def has_dType(self, dType):
        if dType in self.data_types:
            return bool(getattr(self, dType).sites)
        else:
            print('{} is not a valid data type'.format(dType))
            return False

    @utils.enforce_input(sites=list, periods=list, components=list, hTol=float, lTol=float)
    def get_data_from_raw(self, lTol=None, hTol=None, sites=None, periods=None, components=None):
        self.data.sites = self.raw_data.get_data(sites=sites, periods=periods,
                                                 components=components, lTol=lTol,
                                                 hTol=hTol)
        self.data.site_names = self.raw_data.site_names
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

    def write_data(self, outfile='', overwrite=False):
        if outfile == '':
            print('You should probably name your output first...')
            return False
        if not utils.check_file(outfile) or overwrite:
            self.data.write(outfile=outfile)
            return True
        else:
            print('File already exists')
            return False

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

    # Scale error map by how much the period differs, or do a straight error mapping
    # First value is 'scaled' or 'straight', second value is mult. For scaled, this is multiplied by
    # the percent difference between periods. I.E. if the chosen period is 1% away from an existing period,
    # and the map is 10, the error map is 1 * 10 = 10.
    HIGHFREQ_MAP = 20
    OUTLIER_MAP = 10
    XXYY_MAP = 5
    NO_PERIOD_MAP = 50
    NO_COMP_MAP = 9999

    def __init__(self, datafile='', listfile=''):
        """Summary

        Args:
            datafile (str, optional): Description
            listfile (str, optional): Description
        """
        self.datafile = datafile
        self.site_names = []
        self._runErrors = []
        self.periods = []
        self.inv_type = None
        self.__read__(listfile=listfile, datafile=datafile)
        if self.sites:
            self.components = self.sites[self.site_names[0]].components
            # self.locations = {site.name: site.locations for site in self.sites.values()}
            self.locations = self.get_locs()
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
            all_data = WS_io.read_data(datafile=datafile,
                                       site_names=self.site_names,
                                       filetype='data', invType=invType)
            self.sites = {}
            if not self.site_names:
                self.site_names = [str(x) for x in range(1, len(all_data) + 1)]
            self.inv_type = invType
            for site_name, site in all_data.items():
                self.sites.update({site_name:
                                   Site(name=site_name,
                                        data=site['data'],
                                        errors=site['errors'],
                                        errmap=site.get('errmap', None),
                                        periods=site['periods'],
                                        locations=site['locations'],
                                        azimuth=site['azimuth']
                                        )})
            # if not self.site_names:
            #     self.site_names = [site for site in self.sites.keys()]
            self.periods = self.sites[self.site_names[0]].periods
            self.azimuth = self.sites[self.site_names[0]].azimuth
        else:
            self.sites = {}

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
            print('Inversion Type not set. Returning currently set components')
            components = self.components
        elif self.inv_type == 1:
            components = self.ACCEPTED_COMPONENTS[:8]
        elif self.inv_type == 2:
            components = self.ACCEPTED_COMPONENTS[2:5]
        elif self.inv_type == 3:
            components = self.ACCEPTED_COMPONENTS[8:]
        elif self.inv_type == 4:
            components = self.ACCEPTED_COMPONENTS[2:]
        elif self.inv_type == 5:
            components = self.ACCEPTED_COMPONENTS
        return components

    def write(self, outfile):
        WS_io.write_data(data=self, outfile=outfile)

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


class Model(object):
    """Summary
    """

    def __init__(self, modelfile=''):
        self._xCS = []
        self._yCS = []
        self._zCS = []
        self._dx = []
        self._dy = []
        self._dz = []
        self.vals = []
        self.file = modelfile
        self.origin = (0, 0)
        self.UTM_zone = None
        self.__read__(modelfile=modelfile)

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

    def to_vtk(self, outfile=None):
        if not outfile:
            print('Must specify output file name')
            return
        WS_io.model_to_vtk(self, outfile=outfile)

    def is_half_space(self):
        return np.all(np.equal(self.vals.flatten(), self.vals[0, 0, 0]))

    def update_vals(self, old_vals, new_vals, axis):
        if self.is_half_space():
            bg = self.vals[0, 0, 0]
            self.vals = np.ones([self.nx, self.ny, self.nz]) * bg
            return
        else:
            print('Changing mesh on non-half-space model not implemented yet.')

    # Not super necessary but we'll see.
    def insert_mesh(self, dim, idx, val):
        getattr(self, dim).insert(idx, val)

    def generate_mesh(self, site_locs, min_x, min_y, num_pads, pad_mult):
        pass
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
        self._dx = vals
        self._xCS = list(np.diff(self._dx))

    @property
    def dy(self):
        return self._dy

    @dy.setter
    def dy(self, vals):
        self._dy = vals
        self._yCS = list(np.diff(self._dy))

    @property
    def dz(self):
        return self._dz

    @dz.setter
    def dz(self, vals):
        self._dz = vals
        self._zCS = list(np.diff(self._dz))

    @property
    def xCS(self):
        return self._xCS

    @xCS.setter
    def xCS(self, vals):
        self._xCS = vals
        self._dx = list(np.cumsum([0, *vals]) - np.sum(vals) / 2)

    @property
    def yCS(self):
        return self._yCS

    @yCS.setter
    def yCS(self, vals):
        self._yCS = vals
        self._dy = list(np.cumsum([0, *vals]) - np.sum(vals) / 2)

    @property
    def zCS(self):
        return self._zCS

    @zCS.setter
    def zCS(self, vals):
        self._zCS = vals
        self._dz = list(np.cumsum([0, *vals]))


class Response(Data):
    """Summary
    """


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
                 errors={}, errmap=None, azimuth=None, flags=None):
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
        self.errfloorZ = 5
        self.errfloorT = 15
        if errmap:
            self.errmap = errmap
        else:
            self.errmap = self.generate_errmap(mult=1)
        if flags:
            self.flags = flags
        else:
            self.flags = self.generate_errmap(mult=1)
            # self.errmap = {}
            # self.flags = {}
            # for comp in self.data.keys():
            #     self.errmap.update({comp:
            #                         np.ones(np.shape(self.data[comp]))})
        self.used_error = {}
        # Add dummy data to missing components
        self.calculate_used_errors()
        # for comp, val in self.errors.items():
        #     self.used_error.update({comp: val * self.errmap[comp]})

        # Rotate all sites to 0 degrees to start
        # self.rotate_data(azi=0)

    def calculate_used_errors(self, errfloorZ=None, errfloorT=None):
        if errfloorZ is None:
            errfloorZ = self.errfloorZ
        else:
            self.errfloorZ = errfloorZ
        if errfloorT is None:
            errfloorT = self.errfloorT
        else:
            self.errfloorT = errfloorT
        comps = set([comp[:3] for comp in self.components])
        if 'ZXY' in comps:
            ZXY = self.data['ZXYR'] + 1j * self.data['ZXYI']
            ZYX = self.data['ZYXR'] + 1j * self.data['ZYXI']
            erabsz = np.sqrt(np.absolute(ZXY * ZYX)) * errfloorZ / 100
            mapped_err = {comp: self.errors[comp] * self.errmap[comp] for
                          comp in self.errors if comp[0] == 'Z'}
            for comp, err in mapped_err.items():
                self.used_error[comp] = np.array([max(fl, mapped) for (fl, mapped) in zip(erabsz, err)])
        if 'TZX' in comps:
            TZX = self.data['TZXR'] + 1j * self.data['TZXI']
            TZY = self.data['TZYR'] + 1j * self.data['TZYI']
            ERT = np.sqrt(np.absolute(TZX * TZX) + np.absolute(TZY * TZY))
            erabst = ERT * errfloorT / 100
            mapped_err = {comp: self.errors[comp] * self.errmap[comp] for
                          comp in self.errors if comp[0] == 'T'}
            for comp, err in mapped_err.items():
                self.used_error[comp] = np.array([max(fl, mapped) for (fl, mapped) in zip(erabst, err)])

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
        Only adds one period. Lists of periods will have to be handled by outer loops.

        Args:
            data (TYPE): Description
            new_periods (TYPE): Description
            errors (None, optional): Description
            errmap (None, optional): Description
        """
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
                new_data[comp].insert(ind - 1, data[comp][ii])
                new_errors[comp].insert(ind - 1, errors[comp][ii])
                new_errmap[comp].insert(ind - 1, errmap[comp][ii])
                if period not in current_periods:
                    current_periods.insert(ind, period)
        self.data = {comp: utils.list2np(new_data[comp]) for comp in self.components}
        self.errors = {comp: utils.list2np(new_errors[comp]) for comp in self.components}
        self.errmap = {comp: utils.list2np(new_errmap[comp]) for comp in self.components}
        self.periods = utils.list2np(utils.truncate(current_periods))
        self.calculate_used_errors()

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
                # self.used_error[comp][ind] = self.errors[comp][ind] * mult
        self.calculate_used_errors()

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
    def get_data(self, periods=None, components=None,
                 lTol=0.02, hTol=0.10):
        if periods is None:
            periods = self.periods
        if components is None:
            components = self.components
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
                for jj in range(max(1, idx - 2), min(nper, idx + 2)):
                    expected += self.data[comp][jj]
                expected /= jj
                tol = abs(2 * expected)
                diff = datum - expected
                if abs(diff) > tol:
                    self.errmap[comp][idx] *= outlier_map
                    self.flags[comp][idx] = self.OUTLIER_FLAG


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
        if not datpath:
            datpath = os.path.dirname(listfile)
        self.datpath = datpath
        self.listfile = listfile
        if listfile:
            self.__read__(listfile, datpath)
        else:
            self.sites = {}
            self.datpath = ''
            self.site_names = []
            self.locations = []
        self.low_tol = 0.02
        self.high_tol = 0.1
        self.count_tol = 0.5
        self.master_periods = self.master_period_list()
        self.narrow_periods = self.narrow_period_list()
        self.azimuth = 0
        for site_name in self.site_names:
            self.sites[site_name].rotate_data(azi=0)

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
        self.locations = Data.get_locs(self)
        self.datpath = datpath
        self.listfile = listfile

    def remove_components(self, sites=None, components=None):
        for site in sites:
            self.sites[site].remove_components(components=components)

    def check_dummy_data(self, threshold=0.001):
        # for comp in ('TZXR', 'TZXI', 'TZYR', 'TZYI'):
        # Only need to check one, right?
        sites = []
        for site in self.sites.values():
            if 'TZXR' in site.components:
                if np.all(abs(site.data['TZXR']) < threshold):
                    sites.append(site.name)
        return sites

    def get_locs(self, sites=None):
        return Data.get_locs(self, sites)

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
