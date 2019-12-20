import numpy as np
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
import pyMT.utils as utils
from pyMT.IO import debug_print
from pyMT.e_colours import colourmaps as cm
import naturalneighbor as nn
# import matplotlib.pyplot as plt


# ==========TODO=========== #
# This needs to be changed to help speed up re-drawing
# Need to factor out a lot of things so each function does less on its own
# I.E. plot_site should just be responsible for setting data.
# Things like axes.clear() should only be called when they absolutely
# need to. This slows things down a lot.
# Only re-draw / plot what you need to. For changing error bars,
# only the one plot needs to be changed, and only the y-data needs to
# be changed. For adding a period selection, only y-data needs changing.
# The main time that the whole figure would need to be updated is if
# all the plots are being changed (to new sites), or if the number of
# subplots are changing. Even plotting new components can probably be
# handled without clearing all the axes. #
# Also (way in the future) I could consider replacing some groupings of
# attributes with named tuples. E.G. group things like 'mec', 'markersize',
# etc into a 'plot_options' named tuple attribute, so they can be retrieved
# and set all together
# Can simplify the plotting mechanism and remove the need to return max and mins
# Data plotted on an axis can be retrieved after the fact with axes[#].line[#].get_ydata()
# However right now, for some reason each axis has 6 things plotted on it in my testruns
# example, when there should only be 2 (One for each component plotted in the raw data)
# What is happening is that the ax.errorbar method creates 3 lines: The actual plot,
# plus x and y errorbars (even though X is empty). What I need to do then, is find a way
# to differentiate these, and then I can just set ydata on the appropriate ones when
# redrawing axes, rather than having to call plot_site all over again. This should speed
# things up, however I may need a dictionary or something to hold each axes lines (also,
# does errorbar always return the lines in the same order?)
#
# The plotter (or maybe more importantly the data structures) need to be tested for rotation. Generate
# a test dataset using j2ws3d and make sure that the data points are the same when data is chosen and
# rotated. It looks like this is not the case (raw data rotated and plotted is noticeably different)
# from the data that is plotted.
# ========================= #

def format_data_coords(x, y):
    freq = np.log10(1 / 10**x)
    return '\n'.join(['Log10: period={}, frequency={}, y={}',
                      'period={}, frequency={}']).format(utils.truncate(x),
                                                         utils.truncate(freq),
                                                         utils.truncate(y),
                                                         utils.truncate(10**x),
                                                         utils.truncate(10**freq))


class format_model_coords(object):
    def __init__(self, im, X, Y, x_label='y', y_label='y'):
        self.im = im
        self.x_label = x_label
        self.y_label = y_label
        self.X = X
        self.Y = Y

    def __call__(self, x, y):
        # col = int(x + 0.5)
        # row = int(y + 0.5)
        # if col >=0 and col < numcols and row >=0 and row < numrows:
        # val = X[row, col]
        x_idx = (np.abs(self.X - x + 0.5)).argmin() - 1
        y_idx = (np.abs(self.Y - y + 0.5)).argmin() - 1
        # vals = np.reshape(self.im.get_array(), [len(self.X), len(self.Y)])
        vals = np.array(self.im.get_array())
        # print((vals.shape, len(self.X), len(self.Y), len(self.X) * len(self.Y)))
        vals = 10 ** np.reshape(vals, (len(self.Y) - 1, len(self.X) - 1))[y_idx, x_idx]
        # z = vals[x_idx, y_idx]
        # print((x_idx, y_idx))
        # z = self.im.get_array()[x_idx * len(self.X) + y_idx]
        # z = self.im.get_array()[x_idx * len(self.Y) + y_idx]
        # z = self.im.get_array()[y_idx * len(self.X) + x_idx]
        # z = self.im.get_array()[y_idx * len(self.Y) + x_idx]
        # print(dir(self.im))
        # print(z)
        return '\t'.join(['{}: {:>4.4g} {}',
                          '{}: {:>4.4g} {}\n',
                          'Resistivity: {} {}']).format(self.x_label, utils.truncate(self.X[x_idx]), 'km',
                                                        self.y_label, utils.truncate(self.Y[y_idx]), 'km',
                                                        utils.truncate(vals), 'ohm-m')


class DataPlotManager(object):

    def __init__(self, fig=None):
        self.link_axes_bounds = False
        self.axis_padding = 0.1
        self.colour = ['darkgray', 'r', 'g', 'm', 'y', 'lime', 'peru', 'palegreen']
        self.sites = {'raw_data': [], 'data': [], 'response': []}
        self.toggles = {'raw_data': True, 'data': True, 'response': True}
        self.marker = {'raw_data': 'o', 'data': 'oo', 'response': '-'}
        self.errors = 'mapped'
        self.which_errors = ['data', 'raw_data']
        self.components = ['ZXYR']
        self.linestyle = '-'
        self.scale = 'sqrt(periods)'
        self.mec = 'k'
        self.markersize = 5
        self.edgewidth = 2
        self.sites = None
        self.tiling = [0, 0]
        self.show_outliers = True
        self.wrap = 0
        self.outlier_thresh = 2
        self.min_ylim = None
        self.max_ylim = None
        self.artist_ref = {'raw_data': [], 'data': [], 'response': []}
        self.y_labels = {'r': 'Log10 App. Rho', 'z': 'Impedance',
                         't': 'Transfer Function', 'p': 'Phase', 'b': 'Apparent Resistivity'}
        self.pt_units = 'degrees'
        if fig is None:
            self.new_figure()
        else:
            self.fig = fig

    @property
    def site_names(self):
        site_containers = self.sites.values()
        sites = next(site for site in site_containers if site != [])
        return [site.name for site in sites]

    @property
    def num_sites(self):
        return max([len(data) for data in self.sites.values()])

    @property
    def units(self):
        # Might have to add a setter later if I want some way to plot RMS or something
        if self.components[0][0].upper() == 'Z':
            units = 'mV/nT'
        elif self.components[0][0].upper() == 'T':
            units = 'Unitless'
        elif self.components[0][0].upper() == 'R' or self.components[0][0].upper() == 'B':
            units = r'${\Omega}$-m'
        elif self.components[0][0].upper() == 'P':
            units = 'Degrees'
        if self.scale.lower() == 'periods' and not any(
                sub in self.components[0].lower() for sub in ('rho', 'pha')):
            units = ''.join(['s*', units])
        elif self.scale.lower() == 'sqrt(periods)' and not any(
                sub in self.components[0].lower() for sub in ('rho', 'pha', 'bost')):
            units = ''.join([r'$\sqrt{s}$*', units])
        return units

    def new_figure(self):
        # self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig = Figure()
        self.axes = []

    def redraw_axes(self):
        # self.fig.clear()
        self.axes = []
        self.tiling = self.gen_tiling()
        for ii in range(self.num_sites):
            # if ii > 0 and self.link_axes_bounds:
            #     self.axes.append(self.fig.add_subplot(self.tiling[0], self.tiling[1], ii + 1,
            #                                           sharex=self.axes[0], sharey=self.axes[0]))
            # else:
            self.axes.append(self.fig.add_subplot(self.tiling[0], self.tiling[1], ii + 1))

    def gen_tiling(self):
        if self.num_sites < 3:
            tiling = [self.num_sites, 1]
        else:
            s1 = np.floor(np.sqrt(self.num_sites))
            s2 = np.ceil(self.num_sites / s1)
            tiling = [int(s1), int(s2)]
        return tiling

    def refresh(self):
        # The sites loaded in are the basis for the rest of the attributes,
        # so load them based on that list.
        pass

    def replace_sites(self, sites_in, sites_out):
        """
        Replace sites_out (which are currently plotted) with sites_in.
        Arg:
            sites_in (dict): A dictionary of the data types and site objects to be plotted.
                             Format is {data_type: {site_name: site object}}
            sites_out (list): List of strings indicating the sites to be replaced.
        """
        snames = self.site_names
        idx = []
        ns_in = max([len(sites) for sites in sites_in.values()])
        if ns_in > len(sites_out):
            sites_in = sites_in[:len(sites_out)]
        elif len(sites_out) > ns_in:
            sites_out = sites_out[:len(sites_in)]
        for dType, site_list in sites_in.items():
            jj = 0
            if sites_in[dType]:
                for ii, site in enumerate(snames):
                    if site in sites_out:
                        try:
                            self.sites[dType][ii] = sites_in[dType][jj]
                        except IndexError:
                            self.sites[dType].append(sites_in[dType][jj])
                        idx.append(ii)
                        jj += 1
            else:
                self.sites[dType] = []

        # if jj >= len(snames):
        #     self.plot_data()
        # else:
        idx = list(set(idx))
        for ii in idx:
            self.redraw_single_axis(axnum=ii)

    def draw_all(self, sites=None):
        # Refactor plot_data into here. Take out Max and Min stuff and just call
        # set_maxmin() where it will get the data from ydata of the given axes.
        if self.sites is None:
            self.sites = sites
        # Max = np.zeros([tiling[1] * tiling[0]])
        # Min = np.zeros([tiling[1] * tiling[0]])

    def redraw_single_axis(self, site_name='', axnum=None):
        if (site_name == '' and axnum is None):
            print('You have to specify either the axis number or site name...')
            return
        elif axnum is None:
            axnum = self.site_names.index(site_name)
        # print('Checking...')
        # print(axnum, site_name, self.site_names[axnum])
        self.axes[axnum].clear()
        site_name = self.site_names[axnum]
        Max = -99999
        Min = 99999
        for ii, Type in enumerate(['raw_data', 'data', 'response']):
            if self.sites[Type] != []:
                # site = next(s for s in self.sites[Type] if s.name == self.site_names[axnum])
                site = self.sites[Type][axnum]
                if site is not None and self.toggles[Type]:
                    self.axes[axnum], ma, mi, artist = self.plot_site(site, Type=Type,
                                                                      ax=self.axes[axnum])
                    Max = max(Max, ma)
                    Min = min(Min, mi)
                    if axnum >= len(self.artist_ref[Type]):
                        self.artist_ref[Type].append(artist)
                    else:
                        self.artist_ref[Type][axnum] = artist
        if axnum == 0:
            self.set_legend()
        self.set_bounds(Max=Max, Min=Min, axnum=axnum)
        # self.set_bounds(axnum=axnum)
        self.set_labels(axnum, site_name)
        # self.fig.canvas.draw()

    def set_legend(self):
        for ii, comp in enumerate(self.components):
            self.axes[0].plot([], [], color=self.colour[ii], label=comp, marker='o')
        leg = self.axes[0].legend()
        leg.get_frame().set_alpha(0.4)

    def set_labels(self, axnum, site_name):
        self.axes[axnum].set_title(site_name)
        rows = self.tiling[0]
        cols = self.tiling[1]
        if axnum % cols == 0:
            self.axes[axnum].set_ylabel('{} ({})'.format(
                self.y_labels[self.components[0][0].lower()], self.units))
        if axnum + 1 > cols * (rows - 1):
            if 'bost' in self.components[0].lower():
                self.axes[axnum].set_xlabel('log10 Depth (m)')
            else:
                self.axes[axnum].set_xlabel('log10 of Period (s)')

    def plot_data(self, sites=None):
        # print('Don''t use this method anymore, use draw_all instead')
        if self.sites is None:
            self.sites = sites
        tiling = self.gen_tiling()
        self.tiling = tiling
        if self.fig is None:
            print('Opening new figure')
            self.new_figure()
        else:
            self.fig.clear()
        if len(self.fig.get_axes()) != self.num_sites:
            # print('Redrawing Axes')
            self.redraw_axes()
        Max = np.zeros([tiling[1] * tiling[0]])
        Min = np.zeros([tiling[1] * tiling[0]])
        if not self.components:
            self.components = ['ZXYR']
        for jj, Type in enumerate(['raw_data', 'data', 'response']):  # plot raw first, if avail
            for ii, site in enumerate(self.sites[Type]):
                # xi is the row, yi is the column of the current axis.
                if site is not None and self.toggles[Type]:
                    # print('Plotting on axis {} with type {}'.format(ii, Type))
                    self.axes[ii], ma, mi, artist = self.plot_site(site,
                                                                   Type=Type, ax=self.axes[ii])
                    Max[ii] = max(Max[ii], ma)
                    Min[ii] = min(Min[ii], mi)
                    if ii >= len(self.artist_ref[Type]):
                        self.artist_ref[Type].append(artist)
                    else:
                        self.artist_ref[Type][ii] = artist
                if jj == 0:
                    # print(site.name)
                    self.set_labels(axnum=ii, site_name=site.name)
        self.set_bounds(Max=Max, Min=Min, axnum=list(range(ii + 1)))
        # self.set_bounds(axnum=list(range(ii + 1)))
        self.set_legend()
        # plt.show()

    def set_bounds(self, Max, Min, axnum=None):
        # print('I am setting the bounds')
        try:
            for ax in axnum:
                if not self.min_ylim:
                    min_y = Min[ax]
                else:
                    min_y = self.min_ylim
                if not self.max_ylim:
                    max_y = Max[ax]
                else:
                    max_y = self.max_ylim
                axrange = max_y - min_y
                self.axes[ax].set_ylim([min_y - axrange / 4, max_y + axrange / 4])
        except TypeError:
            if not self.min_ylim:
                min_y = Min
            else:
                min_y = self.min_ylim
            if not self.max_ylim:
                max_y = Max
            else:
                max_y = self.max_ylim
            # print(self.min_ylim, self.max_ylim)
            axrange = max_y - min_y
            self.axes[axnum].set_ylim([min_y - axrange / 4, max_y + axrange / 4])
        # if axnum is None:
        #         axnum = range(0, len(self.axes))
        # if self.link_axes_bounds:
        #     self.link_axes(axnum=axnum)
        # else:
        # try:
        #     for ax in axnum:
        #         self.axes[ax].set_ymargin(self.axis_padding)
        #         self.axes[ax].autoscale(self.axis_padding)
        # except TypeError:
        #     self.axes[axnum].set_ymargin(self.axis_padding)
        #     self.axes[axnum].autoscale()

    @utils.enforce_input(axnum=list, x_bounds=list, y_bounds=list)
    def link_axes(self, axnum, x_bounds=None, y_bounds=None):
        if axnum is None:
            axnum = range(0, len(self.axes))
        if not x_bounds:
            x_bounds = (min([ax.get_xbound()[0] for ax in self.axes]),
                        max([ax.get_xbound()[1] for ax in self.axes]))
        if not y_bounds:
            y_bounds = (min([ax.get_ybound()[0] for ax in self.axes]),
                        max([ax.get_ybound()[1] for ax in self.axes]))
        # x_axrange = x_bounds[1] - x_bounds[0]
        # y_axrange = y_bounds[1] - y_bounds[0]
        # print(x_bounds, y_bounds)
        for ax in axnum:
            self.axes[ax].set_xmargin(self.axis_padding)
            self.axes[ax].set_ymargin(self.axis_padding)
            self.axes[ax].set_ylim([y_bounds[0], y_bounds[1]])
            self.axes[ax].set_xlim([x_bounds[0], x_bounds[1]])

    def plot_site(self, site, Type='Data', ax=None, ):
        """Summary

        Args:
            errors (str, optional): Description
            ax (None, optional): Description
            components (list, optional): Description
            scale (str, optional): Description
            marker (str, optional): Description
            linestyle (str, optional): Description

        Returns:
            TYPE: Description
        """
        # Can I pass other keyword args through directly to plt?
        ma = []
        mi = []
        linestyle = ''
        marker = self.marker[Type.lower()]
        edgewidth = 0
        if marker == 'oo':
            marker = 'o'
            edgewidth = self.edgewidth
        if marker == '-':
            marker = ''
            linestyle = self.linestyle
        if not self.components:
            self.components = [site.components[0]]
        if self.errors.lower() == 'raw':
            Err = site.errors
            errtype = 'errors'
        elif self.errors.lower() == 'mapped':
            Err = site.used_error
            errtype = 'used_error'
        elif self.errors.lower() == 'none':
            Err = None
            toplotErr = None
            errtype = 'none'
        if Type.lower() == 'response':
            Err = None
            toplotErr = None
            errtype = 'none'
        if Type.lower() not in self.which_errors:
            Err = None
            toplotErr = None
            errtype = 'none'
        for ii, comp in enumerate(self.components):
            try:
                if 'rho' in comp.lower():
                    toplot, e, log10_e = utils.compute_rho(site, calc_comp=comp, errtype=errtype)
                    ind = np.where(toplot == 0)[0]  # np.where returns a tuple, take first element
                    if ind.size != 0:
                        print('{} has bad points at periods {}'.format(site.name, site.periods[ind]))
                        print('Adjusting values, ignore them.')
                        toplot[ind] = np.max(toplot)
                    toplot = np.log10(toplot)
                    if Type.lower() != 'response' and self.errors.lower() != 'none':
                        toplotErr = log10_e
                elif 'pha' in comp.lower():
                    toplot, e = utils.compute_phase(site,
                                                    calc_comp=comp,
                                                    errtype=errtype,
                                                    wrap=self.wrap)
                    if Type.lower() != 'response' and self.errors.lower() != 'none':
                        toplotErr = e
                elif 'pt' in comp.lower():
                    # If PTs are actually the inverted data, take them directly from the site
                    if comp in site.components:
                        toplot = site.data[comp]
                        e = site.used_error[comp]
                        # debug_print(e, 'PT_debug.log')
                    # Otherwise use the associated PT object
                    else:
                        toplot = np.array([getattr(site.phase_tensors[ii],
                                                   comp.upper())
                                           for ii in range(site.NP)])
                        e = np.array([getattr(site.phase_tensors[ii],
                                              comp.upper() + '_error')
                                      for ii in range(site.NP)])
                        # debug_print(e, 'PT_debug.log')
                    # Convert to degrees
                    if Type.lower() != 'response' and self.errors.lower() != 'none':
                        toplotErr = e
                    else:
                        toplotErr = e * 0
                    if self.pt_units.lower() == 'degrees':
                            # debug_print(toplotErr, 'debug.log')
                        toplot = np.rad2deg(np.arctan(toplot))
                        try:
                            toplotErr = np.rad2deg(np.arctan(toplotErr))
                        except AttributeError:  # If plotting PTs, toplotErr will be None
                            toplotErr = None
                elif 'bost' in comp.lower():
                    toplot, depth = utils.compute_bost1D(site, comp=comp)[:2]
                    toplot = np.log10(toplot)
                else:
                    toplot = site.data[comp]
                    if Err is not None:
                        toplotErr = Err[comp]
                    if self.scale == 'sqrt(periods)':
                        toplot = toplot * np.sqrt(site.periods)
                        if Err:
                            toplotErr = Err[comp] * np.sqrt(site.periods)
                    elif self.scale == 'periods':
                        toplot = toplot * site.periods
                        if Err:
                            toplotErr = Err[comp] * site.periods
                if 'bost' in comp.lower():
                    artist = ax.errorbar(np.log10(depth), toplot, xerr=None,
                                         yerr=None, marker=marker,
                                         linestyle=linestyle, color=self.colour[ii],
                                         mec=self.mec, markersize=self.markersize,
                                         mew=edgewidth, picker=3)
                else:
                    artist = ax.errorbar(np.log10(site.periods), toplot, xerr=None,
                                         yerr=toplotErr, marker=marker,
                                         linestyle=linestyle, color=self.colour[ii],
                                         mec=self.mec, markersize=self.markersize,
                                         mew=edgewidth, picker=3)
                if self.show_outliers:
                    ma.append(max(toplot))
                    mi.append(min(toplot))
                else:
                    if (self.toggles['raw_data'] and Type.lower() == 'raw_data') or \
                       (not self.toggles['raw_data']):
                        showdata = self.remove_outliers(site.periods, toplot)
                        ma.append(max(showdata))
                        mi.append(min(showdata))
                    else:
                        ma.append(0)
                        mi.append(0)
            except KeyError:
                # raise(e)
                artist = ax.text(0, 0, 'No Data')
                ma.append(0)
                mi.append(0)
            # if Type == 'data':
            #     ax.aname = 'data'
            # elif Type == 'raw_data':
            #     ax.aname = 'raw_data'
        ax.format_coord = format_data_coords
        # ax.set_title(site.name)
        return ax, max(ma), min(mi), artist

    def remove_outliers(self, periods, data):
        expected = utils.geotools_filter(periods, data, fwidth=self.outlier_thresh)
        return expected
        nper = len(data)
        inds = []
        # for idx, datum in enumerate(data):
        #     ratio = abs((expected[idx] - datum) / expected[idx])
        #     if ratio >= self.outlier_thresh:
        #         inds.append(idx)
        for idx, datum in enumerate(data):
            expected = 0
            for jj in range(max(1, idx - 2), min(nper, idx + 2)):
                expected += data[jj]
            expected /= jj
            # tol = abs(self.outlier_thresh * expected)
            diff = (datum - expected) / expected
            if abs(diff) > self.outlier_thresh:
                inds.append(idx)
        return np.array([x for (idx, x) in enumerate(data) if idx not in inds])


class MapView(object):
    COORD_SYSTEMS = ('local', 'utm', 'latlong')

    def __init__(self, figure=None, **kwargs):
        self.window = {'figure': None, 'axes': None, 'canvas': None, 'colorbar': None}
        if figure:
            self.window = {'figure': figure, 'axes': figure.axes, 'canvas': figure.canvas, 'colorbar': None}
        self.axis_padding = 0.1
        self.fake_im = []
        self.colour = 'k'
        self.site_names = []
        self.site_data = {'raw_data': [], 'data': [], 'response': []}
        self._active_sites = []
        self.site_locations = {'generic': [], 'active': [], 'all': []}
        self.toggles = {'raw_data': False, 'data': False, 'response': False}
        self.site_marker = 'o'
        self.site_fill = True
        self.site_colour = 'k'
        self.arrow_colours = {'raw_data': 'k', 'data': 'b', 'response': 'r'}
        self.linestyle = '-'
        self.mec = 'k'
        self.markersize = 5
        self.edgewidth = 2
        self._coordinate_system = 'local'
        self.artist_ref = {'raw_data': [], 'data': [], 'response': []}
        self.annotate_sites = 'active'
        self.colourmap = 'jet_plus'
        self.rho_cax = [1, 5]
        self.phase_cax = [0, 90]
        self.diff_cax = [-10, 10]
        self.model_cax = [1, 5]
        self.padding_scale = 5
        self.plot_rms = False
        self.use_colourbar = True
        self.pt_scale = 2
        self.induction_scale = 5
        self.induction_cutoff = 2
        self.induction_error_tol = 0.5
        self.rho_error_tol = 1
        self.phase_error_tol = 30
        self.units = 'm'
        self.mesh = False
        self.linewidth = 0.005
        if figure is None:
            self.new_figure()
        else:
            self.window['figure'] = figure
        self.create_axes()

    @property
    def facecolour(self):
        if self.site_fill:
            return self.site_colour
        else:
            return 'none'

    @property
    def cmap(self):
        return cm.get_cmap(self.colourmap)

    @property
    def generic_sites(self):
        return list(set(self.site_names) - set(self.active_sites))

    @property
    def data(self):
        return self.site_data['data']

    @data.setter
    def data(self, data):
        self.site_data['data'] = data

    @property
    def raw_data(self):
        return self.site_data['raw_data']

    @raw_data.setter
    def raw_data(self, raw_data):
        self.site_data['raw_data'] = raw_data

    @property
    def response(self):
        return self.site_data['response']

    @response.setter
    def response(self, response):
        self.site_data['response'] = response

    @property
    def active_sites(self):
        return self._active_sites

    @active_sites.setter
    def active_sites(self, sites=None):
        if not sites:
            sites = []
        elif isinstance(sites, str):
            sites = [sites]
        self._active_sites = sites
        self.set_locations()
        # self.plot_locations()

    @property
    def coordinate_system(self):
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, coordinate_system):
        if coordinate_system in MapView.COORD_SYSTEMS and coordinate_system != self._coordinate_system:
            if self.verify_coordinate_system(coordinate_system):
                self._coordinate_system = coordinate_system
                generic_sites = list(set(self.site_names) - set(self.active_sites))
                self.site_locations['generic'] = self.get_locations(sites=generic_sites)
                self.site_locations['active'] = self.get_locations(sites=self.active_sites)
                self.site_locations['all'] = self.get_locations(sites=self.site_names)
                self.plot_locations()

    def verify_coordinate_system(self, coordinate_system):
        if coordinate_system.lower() == 'local':
            return True
        elif coordinate_system.lower() == 'utm' or coordinate_system.lower() == 'latlong':
            if self.site_data['raw_data']:
                return True
            else:
                return False

    def set_kwargs(self, kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                setattr(self, key, value)

    def new_figure(self):
        if not self.window['figure']:
            self.window['figure'] = Figure()
            self.create_axes()

    def create_axes(self):
        if not self.window['figure'].axes:
            self.window['axes'] = self.window['figure'].add_subplot(111)
        else:
            self.window['axes'] = self.window['figure'].axes

    def set_locations(self):
        self.site_locations['generic'] = self.get_locations(sites=self.generic_sites)
        self.site_locations['active'] = self.get_locations(sites=self.active_sites)

    def get_locations(self, sites=None, coordinate_system=None):
        if not coordinate_system:
            coordinate_system = self.coordinate_system
        else:
            self.coordinate_system = coordinate_system
        if coordinate_system == 'local':
            azi = self.site_data['data'].azimuth
            check_azi = self.site_data['data'].check_azi()
            locs = self.site_data['data'].get_locs(site_list=sites, azi=azi)
        elif coordinate_system.lower() == 'utm':
            azi = self.site_data['raw_data'].azimuth
            check_azi = self.site_data['raw_data'].check_azi()
            locs = self.site_data['raw_data'].get_locs(sites=sites, mode='utm')
        elif coordinate_system.lower() == 'latlong':
            azi = self.site_data['raw_data'].azimuth
            check_azi = self.site_data['raw_data'].check_azi()
            locs = self.site_data['raw_data'].get_locs(sites=sites, mode='latlong')
        if azi % 360 != check_azi % 360:
            print('Rotating')
            locs = utils.rotate_locs(locs, azi)
        return locs

    def set_axis_limits(self, bounds=None, ax=None):
        if ax is None:
            ax = self.window['axes'][0]
        if bounds is None:
            min_x, max_x = (min(self.site_data['data'].locations[:, 1]),
                            max(self.site_data['data'].locations[:, 1]))
            min_y, max_y = (min(self.site_data['data'].locations[:, 0]),
                            max(self.site_data['data'].locations[:, 0]))
            x_pad = (max_x - min_x) / self.padding_scale
            y_pad = (max_y - min_y) / self.padding_scale
            ax.set_xlim([min_x - x_pad, max_x + x_pad])
            ax.set_ylim([min_y - y_pad, max_y + y_pad])
        else:
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])

    def plot_locations(self):
        # if not self.window['figure']:
        #     print('No figure to plot to...')
        #     return
        marker_size = {'generic': [self.markersize ** 2], 'active': [self.markersize ** 2]}
        if self.plot_rms:
            marker = 'o'
            marker_size['generic'] = np.array([self.dataset.rms['Station'][site]['Total']
                                               for site in self.generic_sites])
            marker_size['active'] = np.array([self.dataset.rms['Station'][site]['Total']
                                              for site in self.active_sites])
            marker_size['generic'] = (utils.normalize(marker_size['generic'],
                                                      lower=1, upper=2, explicit_bounds=True) *
                                      self.markersize) ** 2
            marker_size['active'] = (utils.normalize(marker_size['active'],
                                                     lower=1, upper=2, explicit_bounds=True) *
                                     self.markersize) ** 2
            facecolour = 'None'
        else:
            marker = self.site_marker
            facecolour = self.facecolour
        if len(self.site_locations['generic']) > 0:
            try:
                self.window['axes'][0].scatter(self.site_locations['generic'][:, 1],
                                               self.site_locations['generic'][:, 0],
                                               marker=marker,
                                               s=marker_size['generic'],
                                               edgecolors=self.site_colour,
                                               linewidths=self.edgewidth,
                                               facecolors=facecolour)
            except IndexError:
                debug_print(len(self.site_locations['generic']), 'debug.log')

            # print('Hi Should be plotting now')
            # print(self.site_locations['generic'])
            # print(marker_size['generic'])
            # print(marker)
        if self.active_sites:
            self.window['axes'][0].scatter(self.site_locations['active'][:, 1],
                                           self.site_locations['active'][:, 0],
                                           marker=marker,
                                           s=marker_size['active'],
                                           edgecolors=self.site_colour,
                                           linewidths=self.edgewidth,
                                           facecolors=facecolour)
            if self.annotate_sites == 'active':
                for ii, (xx, yy) in enumerate(self.site_locations['active']):
                    self.window['axes'][0].annotate(self.active_sites[ii], xy=(yy, xx))
            elif self.annotate_sites == 'all':
                for ii, (xx, yy) in enumerate(self.site_locations['active']):
                        self.window['axes'][0].annotate(self.active_sites[ii], xy=(yy, xx))
                for ii, (xx, yy) in enumerate(self.site_locations['generic']):
                    self.window['axes'][0].annotate(self.generic_sites[ii], xy=(yy, xx))
        self.set_axis_limits()

    @utils.enforce_input(data_type=list, normalize=bool, period_idx=int)
    def plot_induction_arrows(self, data_type='data', normalize=True, period_idx=1):
        # max_length = np.sqrt((np.max(self.site_locations['all'][:, 0]) -
        #                       np.min(self.site_locations['all'][:, 0])) ** 2 +
        #                      (np.max(self.site_locations['all'][:, 1]) -
        #                       np.min(self.site_locations['all'][:, 1])) ** 2) / 10
        x_max, x_min = (np.max(self.site_locations['all'][:, 0]), np.min(self.site_locations['all'][:, 0]))
        y_max, y_min = (np.max(self.site_locations['all'][:, 1]), np.min(self.site_locations['all'][:, 1]))
        max_length = np.max((x_max - x_min, y_max - y_min))

        for dType in data_type:
            X, Y = [], []
            if dType.lower() == 'data':
                colour = 'k'
            elif dType.lower() == 'raw_data':
                colour = 'k'
            elif dType.lower() == 'response':
                colour = 'r'
            idx = []
            for ii, site in enumerate(self.site_names):
                # Just takes the last frequency. Will have to grab the right one from a list.
                # if ((abs(self.site_data[dType].sites[site].used_error['TZXR'][period_idx] /
                #          self.site_data[dType].sites[site].data['TZXR'][period_idx]) < self.induction_error_tol) and
                #     (abs(self.site_data[dType].sites[site].used_error['TZYR'][period_idx] /
                #          self.site_data[dType].sites[site].data['TZYR'][period_idx]) < self.induction_error_tol)):
                # if True:
                #     idx.append(ii)
                #     if 'TZXR' in self.site_data[dType].sites[site].components:
                #         X.append(-self.site_data[dType].sites[site].data['TZXR'][period_idx])
                #     else:
                #         X.append(0)
                #     if 'TZYR' in self.site_data[dType].sites[site].components:
                #         Y.append(-self.site_data[dType].sites[site].data['TZYR'][period_idx])
                #     else:
                #         Y.append(0)
                idx.append(ii)
                if 'TZXR' in self.site_data[dType].sites[site].components:
                    X.append(-self.site_data[dType].sites[site].data['TZXR'][period_idx])
                else:
                    X.append(0)
                if 'TZYR' in self.site_data[dType].sites[site].components:
                    Y.append(-self.site_data[dType].sites[site].data['TZYR'][period_idx])
                else:
                    Y.append(0)
            if idx:
                arrows = np.transpose(np.array((X, Y)))
                # arrows = utils.normalize_arrows(arrows)
                lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                largest_arrow = np.max(lengths)
                lengths[lengths == 0] = 1
                arrows[lengths > 1, 0] = 2 * arrows[lengths > 1, 0] / lengths[lengths > 1]
                arrows[lengths > 1, 1] = 2 * arrows[lengths > 1, 1] / lengths[lengths > 1]
                lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                cutoff_idx = lengths > self.induction_cutoff
                # print(idx)
                arrows[cutoff_idx, :] = 0
                lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                largest_arrow = np.max(lengths)
                # print('Hello I am inside')
                # print(normalize)
                if normalize:
                    arrows = 0.5 * arrows / np.transpose(np.tile(lengths, [2, 1]))
                    # print('Normalizing...')
                # else:
                    # arrows = max_length * arrows / (largest_arrow * 1000)
                    # arrows = arrows / np.max(lengths)
                    # arrows *= max_length / 50000
                    # print('Shrinking arrows')
                #     print('Not normalizing')
                arrows = arrows * self.induction_scale / (50) # * np.sqrt(lengths))
                # print(self.induction_scale)
                # print(arrows)
                # lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                # print(self.site_locations['all'])
                # headwidth = max(3 * self.induction_scale / 5, 1)
                # width = 0.0025 * self.induction_scale / 5
                headwidth = max(3, 1)
                # scale = self.induction_scale * max_length / 100
                lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                largest_arrow = np.max(lengths)
                # print(largest_arrow)
                width = 0.0025
                self.window['axes'][0].quiver(self.site_locations['all'][idx, 1],
                                              self.site_locations['all'][idx, 0],
                                              arrows[:, 1],
                                              arrows[:, 0],
                                              color=colour,
                                              headwidth=headwidth,
                                              width=width,
                                              zorder=10,
                                              scale=1,
                                              scale_units=None)
                self.set_axis_limits()

    @utils.enforce_input(data_type=list, normalize=bool, fill_param=str, period_idx=int)
    def plot_phase_tensor(self, data_type='data', normalize=True, fill_param='Beta', period_idx=1):
        def generate_ellipse(phi):
            jx = np.cos(np.arange(0, 2 * np.pi, np.pi / 30))
            jy = np.sin(np.arange(0, 2 * np.pi, np.pi / 30))
            phi_x = phi[0, 0] * jx + phi[0, 1] * jy
            phi_y = phi[1, 0] * jx + phi[1, 1] * jy
            return phi_x, phi_y
        ellipses = []
        fill_vals = []
        if fill_param != 'Lambda':
            fill_param = fill_param.lower()
        if len(data_type) == 2:
            data_type = ['data', 'response']
        X_all, Y_all = self.site_locations['all'][:, 0], self.site_locations['all'][:, 1]
        scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
                        (np.max(Y_all) - np.min(Y_all)) ** 2)
        good_idx = []
        for ii, site_name in enumerate(self.site_names):
            site = self.site_data[data_type[0]].sites[site_name]
            if len(data_type) == 1:
                phase_tensor = site.phase_tensors[period_idx]
                # if ((phase_tensor.rhoxy_error / phase_tensor.rhoxy < self.rho_error_tol) and
                #     (phase_tensor.rhoyx_error / phase_tensor.rhoyx < self.rho_error_tol) and
                #     (phase_tensor.phasexy_error < self.phase_error_tol) and
                #     (phase_tensor.phaseyx_error < self.phase_error_tol)):
                if True:
                    cont = 1
                    phi_x, phi_y = generate_ellipse(phase_tensor.phi)
                    norm_x, norm_y = (phi_x, phi_y)
                    good_idx.append(ii)
                    # print('Site: {} Phase error: {}, Rho error {}'.format(site_name, phase_tensor.phaseyx_error, phase_tensor.rhoyx_error))
                else:
                    cont = 0
                    # print('Site: {} Phase error: {}, Rho error {}'.format(site_name, phase_tensor.phaseyx_error, phase_tensor.rhoyx_error))
                    phi_x, phi_y = generate_ellipse(phase_tensor.phi)
                    norm_x, norm_y = (phi_x, phi_y)
            else:
                if True:  # Buffer for error tolerances at some point...
                    phase_tensor = (self.site_data[data_type[0]].sites[site_name].phase_tensors[period_idx] -
                                    self.site_data[data_type[1]].sites[site_name].phase_tensors[period_idx])
                    good_idx.append(ii)
                    phi_x, phi_y = generate_ellipse(phase_tensor.phi)
                    norm_x, norm_y = generate_ellipse(self.site_data[data_type[0]].sites[site_name].phase_tensors[period_idx].phi)
                    cont = 1
            if cont:
                X, Y = X_all[ii], Y_all[ii]
                phi_x, phi_y = (1000 * phi_x / phase_tensor.phi_max,
                                1000 * phi_y / phase_tensor.phi_max)
                radius = np.max(np.sqrt(phi_x ** 2 + phi_y ** 2))
                # if radius > 1000:
                phi_x, phi_y = [(self.pt_scale * scale / (radius * 100)) * x for x in (phi_x, phi_y)]
                ellipses.append([Y - phi_x, X - phi_y])
                fill_vals.append(getattr(phase_tensor, fill_param))
        fill_vals = np.array(fill_vals)
        if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3']:
            lower, upper = (0, 90)
        elif fill_param in ['Lambda']:
            lower, upper = (0, 1)
        elif fill_param == 'beta':
            lower, upper = (-10, 10)
        elif fill_param in ['alpha', 'azimuth']:
            lower, upper = (-90, 90)
        elif fill_param in ('delta'):
            lower, upper = (0, 100)
        if fill_param not in ('delta', 'Lambda'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        # fill_vals = utils.normalize(fill_vals,
        #                             lower=lower,
        #                             upper=upper,
        #                             explicit_bounds=True)
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        for ii, ellipse in enumerate(ellipses):
            self.window['axes'][0].fill(ellipse[0], ellipse[1],
                                        color=self.cmap(norm_vals[ii]),
                                        zorder=0,
                                        edgecolor='k',
                                        linewidth=2)
            self.window['axes'][0].plot(ellipse[0], ellipse[1],
                                        'k-', linewidth=1)
        fake_vals = np.linspace(lower, upper, len(fill_vals))
        self.fake_im = self.window['axes'][0].scatter(self.site_locations['all'][good_idx, 1],
                                                      self.site_locations['all'][good_idx, 0],
                                                      c=fake_vals, cmap=self.cmap)
        # fake_im.colorbar()
        self.fake_im.set_visible(False)
        if not self.window['colorbar'] and self.use_colourbar:
            self.window['colorbar'] = self.window['figure'].colorbar(mappable=self.fake_im)
            label = self.get_label(fill_param)
            self.window['colorbar'].set_label(label,
                                              rotation=270,
                                              labelpad=20,
                                              fontsize=18)
        self.set_axis_limits()

    @utils.enforce_input(data_type=list, normalize=bool, fill_param=str, period_idx=int)
    def plot_phase_bar(self, data_type='data', normalize=True, fill_param='Beta', period_idx=1):
        rectangles = []
        fill_vals = []
        if fill_param != 'Lambda':
            fill_param = fill_param.lower()
        if len(data_type) == 2:
            data_type = ['data', 'response']
        X_all, Y_all = self.site_locations['all'][:, 0], self.site_locations['all'][:, 1]
        scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
                        (np.max(Y_all) - np.min(Y_all)) ** 2)
        good_idx = []
        for ii, site_name in enumerate(self.site_names):
            site = self.site_data[data_type[0]].sites[site_name]
            phase_tensor = site.phase_tensors[period_idx]
            xy = [Y_all[ii], X_all[ii]]
            width = phase_tensor.phi_max / 5
            height = 2 * phase_tensor.phi_min / phase_tensor.phi_max
            width = 0.2
            # height = 1
            width, height = [(self.pt_scale * scale) * x / 100 for x in (width, height)]
            xy[0] = xy[0] + height * (np.sin(phase_tensor.azimuth)) / 2 - width * (np.cos(phase_tensor.azimuth)) / 2
            xy[1] = xy[1] - height * (np.cos(phase_tensor.azimuth)) / 2 - width * (np.sin(phase_tensor.azimuth)) / 2
            rectangles.append((xy, width, height, np.rad2deg(phase_tensor.azimuth)))
            #     ellipses.append([Y - phi_x, X - phi_y])
            fill_vals.append(getattr(phase_tensor, fill_param))
        fill_vals = np.array(fill_vals)
        if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3']:
            lower, upper = (0, 90)
        elif fill_param in ['Lambda']:
            lower, upper = (0, 1)
        elif fill_param == 'beta':
            lower, upper = (-6, 6)
        elif fill_param in ['alpha', 'azimuth']:
            lower, upper = (-90, 90)
        elif fill_param in ('delta'):
            lower, upper = (0, 100)
        if fill_param not in ('delta', 'Lambda'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        # print(['XY: ', [x[0] for x in rectangles]])
        # print(['width: ', [x[1] for x in rectangles]])
        # print(['height: ', [x[2] for x in rectangles]])
        # print(['angle: ', [x[3] for x in rectangles]])
        # print(['Fill:', [x for x in norm_vals]])
        for ii, rectangle in enumerate(rectangles):
            rect = patches.Rectangle(xy=rectangle[0],
                                     width=rectangle[1],
                                     height=rectangle[2],
                                     angle=rectangle[3],
                                     color=self.cmap(norm_vals[ii]))
            self.window['axes'][0].add_patch(rect)
        self.set_axis_limits()

    @utils.enforce_input(data_type=list, normalize=bool, fill_param=str, period_idx=int)
    def plot_phase_bar2(self, data_type='data', normalize=True, fill_param='Beta', period_idx=1):
        rectangles = []
        fill_vals = []
        if fill_param != 'Lambda':
            fill_param = fill_param.lower()
        if len(data_type) == 2:
            data_type = ['data', 'response']
        X_all, Y_all = self.site_locations['all'][:, 0], self.site_locations['all'][:, 1]
        scale = np.sqrt((np.max(X_all) - np.min(X_all)) ** 2 +
                        (np.max(Y_all) - np.min(Y_all)) ** 2)
        good_idx = []
        for ii, site_name in enumerate(self.site_names):
            site = self.site_data[data_type[0]].sites[site_name]
            phase_tensor = site.phase_tensors[period_idx]
            xy = [Y_all[ii], X_all[ii]]
            width = phase_tensor.phi_max / 5
            height = 2 * phase_tensor.phi_min / phase_tensor.phi_max
            width = 0.2
            # height = 1
            width, height = [(self.pt_scale * scale) * x / 100 for x in (width, height)]
            # xy[0] -= height * (np.sin(phase_tensor.azimuth)) / 2 + width * (np.cos(phase_tensor.azimuth)) / 2
            # xy[1] -= height * (np.cos(phase_tensor.azimuth)) / 2 + width * (np.sin(phase_tensor.azimuth)) / 2
            rectangles.append((xy, width, height, np.rad2deg(phase_tensor.azimuth)))
            #     ellipses.append([Y - phi_x, X - phi_y])
            fill_vals.append(getattr(phase_tensor, fill_param))
        fill_vals = np.array(fill_vals)
        if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3']:
            lower, upper = (0, 90)
        elif fill_param in ['Lambda']:
            lower, upper = (0, 1)
        elif fill_param == 'beta':
            lower, upper = (-6, 6)
        elif fill_param in ['alpha', 'azimuth']:
            lower, upper = (-90, 90)
        elif fill_param in ('delta'):
            lower, upper = (0, 100)
        if fill_param not in ('delta', 'Lambda'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        for ii, rectangle in enumerate(rectangles):
            rect = patches.Rectangle(xy=rectangle[0],
                                     width=rectangle[1],
                                     height=rectangle[2],
                                     angle=rectangle[3],
                                     color='k',
                                     edgecolor='k')
            self.window['axes'][0].add_patch(rect)
        self.set_axis_limits()


    @utils.enforce_input(data_type=list, fill_param=str, n_interp=int, period_idx=int)
    def plan_pseudosection(self, data_type='data', fill_param='rhoxy', n_interp=200, period_idx=0):
        vals = []
        #  Just make sure data is the first one
        if len(data_type) == 2:
            data_type = ['data', 'response']
        for ii, dt in enumerate(data_type):
            data = self.site_data[dt]
            # temp_vals = []
            # good_idx = []
            if 'rho' in fill_param.lower():
                # if ((phase_tensor.rhoxy_error / phase_tensor.rhoxy < self.rho_error_tol) and
                #     (phase_tensor.rhoyx_error / phase_tensor.rhoyx < self.rho_error_tol) and
                #     (phase_tensor.phasexy_error < self.phase_error_tol) and
                #     (phase_tensor.phaseyx_error < self.phase_error_tol)):
                # for site in data.site_names:
                    # phase_tensor = data.sites[site].phase_tensors[period_idx]
                    # if ((phase_tensor.rhoxy_error / phase_tensor.rhoxy < self.rho_error_tol) and
                    #     (phase_tensor.rhoyx_error / phase_tensor.rhoyx < self.rho_error_tol)):
                    #     temp_vals.append(np.log10(utils.compute_rho(site=data.sites[site],
                    #                                                 calc_comp=fill_param,
                    #                                                 errtype='none')[0][period_idx]))
                    #     good_idx.append(ii)
                vals.append([np.log10(utils.compute_rho(site=data.sites[site],
                                                        calc_comp=fill_param,
                                                        errtype='none')[0][period_idx]) for site in data.site_names])
                # vals.append(temp_vals)
            elif 'pha' in fill_param.lower():
                # vals.append(utils.compute_phase(site=data.sites[site],
                #                                 calc_comp=fill_param,
                #                                 errtype='none',
                #                                 wrap=1)[0][period_idx])
                vals.append([utils.compute_phase(site=data.sites[site],
                                                 calc_comp=fill_param,
                                                 errtype='none',
                                                 wrap=1)[0][period_idx] for site in data.site_names])
        if len(vals) == 1:
            vals = np.array(vals[0])
            diff = False
        else:
            diff = True
            #  Should be response - data so a larger response gives a +ve percent difference
            if 'rho' in fill_param.lower():
                vals = 100 * (np.array(vals[1]) - np.array(vals[0])) / np.array(vals[0])
            elif 'pha' in fill_param.lower():
                vals = (np.array(vals[1]) - np.array(vals[0]))  # / np.array(vals[0])
        loc_x, loc_y = (data.locations[:, 1], data.locations[:, 0])
        loc_z = np.zeros(loc_x.shape)
        points = np.transpose(np.array((loc_x, loc_y, loc_z)))
        min_x, max_x = (min(loc_x), max(loc_x))
        x_pad = (max_x - min_x) / self.padding_scale
        min_x, max_x = (min_x - x_pad, max_x + x_pad)
        min_y, max_y = (min(loc_y), max(loc_y))
        y_pad = (max_y - min_y) / self.padding_scale
        min_y, max_y = (min_y - y_pad, max_y + y_pad)
        # step_size_x = (max_x - min_x) / n_interp
        # step_size_y = (max_y - min_y) / n_interp
        grid_x, grid_y = np.meshgrid(np.linspace(min_x, max_x, n_interp),
                                     np.linspace(min_y, max_y, n_interp))
        grid_ranges = [[min_x, max_x, n_interp * 1j],
                       [min_y, max_y, n_interp * 1j],
                       [0, 1, 1]]
        grid_vals = np.squeeze(nn.griddata(points, vals, grid_ranges))
        if diff and 'rho' in fill_param.lower():
            cax = self.diff_cax
            fill_param = '% Difference'
        elif diff and 'pha' in fill_param.lower():
            cax = self.diff_cax
            fill_param = r'Difference ($^{\circ}$)'
        elif 'rho' in fill_param.lower():
            # grid_vals = np.log10(grid_vals)
            cax = self.rho_cax
        else:
            cax = self.phase_cax
        im = self.window['axes'][0].pcolor(grid_x, grid_y, grid_vals.T,
                                           cmap=self.cmap,
                                           vmin=cax[0], vmax=cax[1],
                                           zorder=0)
        if self.window['colorbar']:
            self.window['colorbar'].remove()
        self.window['colorbar'] = self.window['figure'].colorbar(mappable=im)
        self.window['colorbar'].set_label(fill_param,
                                          rotation=270,
                                          labelpad=20,
                                          fontsize=18)
        self.set_axis_limits()

    def get_label(self, param):
        if param.lower() == 'alpha':
            label = r'$\alpha$'
        elif param.lower() == 'beta':
            label = r'$\beta$'
        elif param.lower() == 'phi_1':
            label = r'$\phi_1$'
        elif param.lower() == 'phi_2':
            label = r'$\phi_2$'
        elif param.lower() == 'phi_3':
            label = r'$\phi_3$'
        elif param.lower() == 'det_phi':
            label = r'$det \phi$'
        elif param.lower() == 'azimuth':
            label = r'Azimuth'
        elif param.lower() == 'phi_max':
            label = r'$\phi_max$'
        elif param.lower() == 'phi_min':
            label = r'$\phi_min$'
        else:
            label = ''
        return label

    def plot_plan_view(self, ax=None, z_slice=0):
        if not ax:
            ax = self.window['axes'][0]
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        X = np.array(self.model.dy)
        Y = np.array(self.model.dx)
        im = ax.pcolormesh(X,
                           Y,
                           np.log10(self.model.vals[:, :, z_slice]),
                           cmap=self.cmap,
                           vmin=self.model_cax[0], vmax=self.model_cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        ax.set_ylabel('Northing (km)')
        ax.format_coord = format_model_coords(im,
                                              X=X, Y=Y,
                                              x_label='Easting', y_label='Northing')

    def plot_x_slice(self, ax=None, x_slice=0):
        if not ax:
            ax = ax
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        X = np.array(self.model.dy)
        Y = np.array(self.model.dz)
        im = ax.pcolormesh(X,
                           Y,
                           np.squeeze(np.log10(self.model.vals[x_slice, :, :])).T,
                           cmap=self.cmap,
                           vmin=self.model_cax[0], vmax=self.model_cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Depth (km)')
        ax.format_coord = format_model_coords(im,
                                              X=X, Y=Y,
                                              x_label='Easting', y_label='Depth')

    def plot_y_slice(self, ax=None, y_slice=0, orientation='xz'):
        if orientation.lower() == 'xz':
            X = np.array(self.model.dx)
            Y = np.array(self.model.dz)
            to_plot = np.squeeze(np.log10(self.model.vals[:, y_slice, :])).T
            x_label = 'Northing (km)'
            y_label = 'Depth (km)'
        else:
            Y = np.array(self.model.dx)
            X = np.array(self.model.dz)
            to_plot = np.squeeze(np.log10(self.model.vals[:, y_slice, :]))
            y_label = 'Northing (km)'
            x_label = 'Depth (km)'
        if not ax:
            ax = ax
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        im = ax.pcolormesh(X, Y, to_plot,
                           cmap=self.cmap,
                           vmin=self.model_cax[0], vmax=self.model_cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.format_coord = format_model_coords(im, X=X, Y=Y,
                                              x_label=x_label.split()[0],
                                              y_label=y_label.split()[0])

