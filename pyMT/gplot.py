import numpy as np
from matplotlib.figure import Figure
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import griddata
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
import pyMT.utils as utils
from pyMT.IO import debug_print
from pyMT.e_colours import colourmaps as cm
from copy import deepcopy
try:
    import naturalneighbor as nn
    has_nn = True
except ModuleNotFoundError:
    has_nn = False
    print('Natural Neighbor not found and will not be available')

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
    def __init__(self, im, X, Y, x_label='y', y_label='y', use_log=True, data_label='Resistivity'):
        self.im = im
        self.x_label = x_label
        self.y_label = y_label
        self.X = X
        self.Y = Y
        self.use_log = use_log
        self.data_label = data_label

    def __call__(self, x, y):
        # col = int(x + 0.5)
        # row = int(y + 0.5)
        # if col >=0 and col < numcols and row >=0 and row < numrows:
        # val = X[row, col]
        for ix, xx in enumerate(self.X):
            if xx > x:
                x_idx = min(ix, len(self.X) - 1) - 1
                break
            x_idx = len(self.X) - 2
        for iy, yy in enumerate(self.Y):
            if yy > y:
                y_idx = min(iy, len(self.Y) - 1) - 1
                break
            y_idx = len(self.Y) - 2
        # x_idx = (np.abs(self.X - x + 0.05)).argmin() - 1
        # y_idx = (np.abs(self.Y - y + 0.05)).argmin() - 1
        # vals = np.reshape(self.im.get_array(), [len(self.X), len(self.Y)])
        vals = np.array(self.im.get_array())
        vals = np.reshape(vals, (len(self.Y) - 1, len(self.X) - 1))[y_idx, x_idx]
        if self.use_log:
            vals = 10 ** vals
        if self.data_label.lower() == 'resistivity':
            self.data_units = 'ohm-m'
        elif self.data_label.lower() in ('phase', 'skew', 'azimuth'):
            self.data_units = 'Degrees'
        elif self.data_label.lower() == 'depth':
            self.data_units = 'km'
        else:
            self.data_units = ''
        # z = vals[x_idx, y_idx]
        # z = self.im.get_array()[x_idx * len(self.X) + y_idx]
        # z = self.im.get_array()[x_idx * len(self.Y) + y_idx]
        # z = self.im.get_array()[y_idx * len(self.X) + x_idx]
        # z = self.im.get_array()[y_idx * len(self.Y) + x_idx]
        return '\t'.join(['{}: {:>4.4g} {}',
                          '{}: {:>4.4g} {}\n',
                          '{}: {} {}']).format(self.x_label, utils.truncate(self.X[x_idx]), 'km',
                                                        self.y_label, utils.truncate(self.Y[y_idx]), 'km',
                                                        self.data_label,
                                                        utils.truncate(vals), self.data_units)


class DataPlotManager(object):

    def __init__(self, fig=None):
        self.link_axes_bounds = False
        self.axis_padding = 0.1
        self.use_designated_colours = True
        self.colour = ['darkgray', 'r', 'royalblue', 'g', 'm', 'y', 'lime', 'peru']
        self.designated_colours = {'xy': 'royalblue',
                                   'yx': 'red',
                                   'xx': 'lime',
                                   'yy': 'peru',
                                   'tzx': 'royalblue',
                                   'tzy': 'red',
                                   'aav': 'green',
                                   'gav': 'magenta',
                                   'det': 'darkgray',
                                   'ssq': 'orange'}
        self.sites = {'raw_data': [], 'data': [], 'response': [], 'smoothed_data': []}
        self.toggles = {'raw_data': True, 'data': True, 'response': True, '1d': False, 'smoothed_data': False}
        self.site1D = []
        ## Modify so real and imaginary use different markers?
        self.marker = {'raw_data': 'o', 'data': 'oo', 'response': '-', '1d': '--', 'smoothed_data': '--'}
        self.errors = 'mapped'
        self.which_errors = ['data', 'raw_data']
        self.components = ['ZXYR']
        self.linestyle = '-'
        self.scale = 'sqrt(periods)'
        self.mec = 'k'
        self.markersize = 5
        self.edgewidth = 2
        self.label_fontsize = 14
        # self.sites = None
        self.tiling = [0, 0]
        self.show_outliers = True
        self.plot_flagged_data = True
        self.wrap_phase = 1
        self.outlier_thresh = 2
        self.min_ylim = None
        self.max_ylim = None
        self.ax_lim_dict = {'rho': [0, 5], 'phase': [0, 120], 'impedance': [-1, 1],
                            'tipper': [-1, 1], 'beta': [-10, 10], 'azimuth': [-90, 90]}
        self.artist_ref = {'raw_data': [], 'data': [], 'response': [], '1d': [], 'smoothed_data': []}
        self.y_labels = {'rho': 'Log10 App. Rho',
                         'zxx': 'Impedance', 'zxy': 'Impedance', 'zyx': 'Impedance', 'zyy': 'Impedance',
                         'tzx': 'Magnitude', 'tzy': 'Magnitude',
                         'bos': 'Apparent Resistivity',
                         'ptx': 'Phase', 'pty': 'Phase', 'phi': 'Phase', 'pha': 'Phase',
                         'bet': 'Skew',
                         'azi': 'Azimuth'}
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
            # units = 'mV/nT'
            units = 'Ohm'
        elif self.components[0][0].upper() == 'T':
            units = 'Unitless'
        elif self.components[0][0].upper() == 'R' or self.components[0][0:4].upper() == 'BOST':
            units = r'${\Omega}$-m'
        elif self.components[0][0].upper() == 'P' or self.components[0][0:4].upper() in ('BETA', 'AZIM'):
            units = 'Degrees'
        if self.scale.lower() == 'periods' and not any(
                sub in self.components[0].lower() for sub in ('rho', 'pha')):
            units = ''.join(['s*', units])
        elif self.scale.lower() == 'sqrt(periods)' and not any(
                sub in self.components[0].lower() for sub in ('rho', 'pha', 'bost')):
            units = ''.join([r'$\sqrt{s}$*', units])
        return units

    def get_designated_colour(self, component, ii):
        component = component.lower()
        if self.use_designated_colours:
            for key in self.designated_colours.keys():
                if key in component:
                    return self.designated_colours[key]
            else:
                return self.colour[ii]
        
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
        # debug_print([sites_out, sites_in], 'debug.log')
        for dType, site_list in sites_in.items():
            jj = 0
            if sites_in[dType]:
                for ii, site in enumerate(snames):
                    if site in sites_out:
                        try:
                            # debug_print([ii, jj, dType], 'debug.log')
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
        self.axes[axnum].clear()
        site_name = self.site_names[axnum]
        Max = -99999
        Min = 99999
        for ii, Type in enumerate(['raw_data', 'data', 'response', '1d', 'smoothed_data']):
            if self.sites[Type] != []:
                # site = next(s for s in self.sites[Type] if s.name == self.site_names[axnum])
                try:
                    site = self.sites[Type][axnum]
                except IndexError:
                    site = None
                if site is not None and self.toggles[Type]:
                    if Type != '1d':
                        self.axes[axnum], ma, mi, artist = self.plot_site(site, Type=Type,
                                                                          ax=self.axes[axnum])
                        Max = max(Max, ma)
                        Min = min(Min, mi)
                        if axnum >= len(self.artist_ref[Type]):
                            self.artist_ref[Type].append(artist)
                        else:
                            self.artist_ref[Type][axnum] = artist
                    else:
                        self.axes[axnum], ma, mi, artist = self.plot_site(self.site1D, Type='1d',
                                                                          ax=self.axes[axnum])
        if axnum == 0:
            self.set_legend()
        self.set_bounds(Max=Max, Min=Min, axnum=axnum)
        # self.set_bounds(axnum=axnum)
        self.set_labels(axnum, site_name)
        # self.fig.canvas.draw()

    def set_legend(self):
        for ii, comp in enumerate(self.components):
            if 'i' in comp.lower():
                marker = 'v'
            else:
                marker = 'o'
            self.axes[0].plot([], [], color=self.get_designated_colour(comp, ii), label=comp, marker=marker)
        leg = self.axes[0].legend()
        leg.get_frame().set_alpha(0.4)

    def set_labels(self, axnum, site_name):
        self.axes[axnum].set_title(site_name)
        rows = self.tiling[0]
        cols = self.tiling[1]
        if axnum % cols == 0:
            self.axes[axnum].set_ylabel('{} ({})'.format(
                self.y_labels[self.components[0][0:3].lower()], self.units), fontsize=self.label_fontsize)
        if axnum + 1 > cols * (rows - 1):
            if 'bost' in self.components[0].lower():
                self.axes[axnum].set_xlabel('log10 Depth (km)', fontsize=self.label_fontsize)
            else:
                self.axes[axnum].set_xlabel('log10 of Period (s)', fontsize=self.label_fontsize)

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
            self.redraw_axes()
        Max = np.zeros([tiling[1] * tiling[0]])
        Min = np.zeros([tiling[1] * tiling[0]])
        if not self.components:
            self.components = ['ZXYR']
        for jj, Type in enumerate(['raw_data', 'data', 'response', '1d', 'smoothed_data']):  # plot raw first, if avail
            if Type == '1d':
                pass
            else:
                for ii, site in enumerate(self.sites[Type]):
                    # xi is the row, yi is the column of the current axis.
                    if site is not None and self.toggles[Type]:
                        self.axes[ii], ma, mi, artist = self.plot_site(site,
                                                                       Type=Type, ax=self.axes[ii])
                        Max[ii] = max(Max[ii], ma)
                        Min[ii] = min(Min[ii], mi)
                        if ii >= len(self.artist_ref[Type]):
                            self.artist_ref[Type].append(artist)
                        else:
                            self.artist_ref[Type][ii] = artist
                    if jj == 0:
                        self.set_labels(axnum=ii, site_name=site.name)
        self.set_bounds(Max=Max, Min=Min, axnum=list(range(ii + 1)))
        # self.set_bounds(axnum=list(range(ii + 1)))
        self.set_legend()
        # plt.show()

    def set_bounds(self, Max, Min, axnum=None):
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
        def pop_flagged_data(x, y, toplotErr, site, component):
            if component in site.components:
                e = site.used_error[component]
            else:
                e = site.used_error[site.components[0]]
            idx = []
            for ii, ie in enumerate(e):
                if ie == site.REMOVE_FLAG:
                    idx.append(ii)
                    # x_popped.append(x.pop(0))
                    # y_popped.append(y.pop(0))
                    # e_popped.append(e.pop(0))
            if idx:
                x = np.delete(x, idx)
                y = np.delete(y, idx)
                if toplotErr is not None:
                    toplotErr = np.delete(toplotErr, idx)
            return x, y, toplotErr  #, x_popped, y_popped, e_popped
        response_types = ('response', '1d')
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
        if marker == '--':
            marker = ''
            linestyle = '--'
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
        if Type.lower() in response_types:
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
                    if Type.lower() not in response_types and self.errors.lower() != 'none':
                        toplotErr = log10_e
                elif 'pha' in comp.lower():
                    toplot, e = utils.compute_phase(site,
                                                    calc_comp=comp,
                                                    errtype=errtype,
                                                    wrap=self.wrap_phase)
                    if Type.lower() not in response_types and self.errors.lower() != 'none':
                        toplotErr = e
                elif 'pt' in comp.lower() or 'phi' in comp.lower() or 'beta' in comp.lower() or 'azimuth' in comp.lower():
                    # If PTs are actually the inverted data, take them directly from the site
                    if comp in site.components:
                        toplot = site.data[comp]
                        e = site.used_error[comp]
                    # Otherwise use the associated PT object
                    elif 'pt' in comp.lower():
                        toplot = np.array([getattr(site.phase_tensors[ii],
                                                   comp.upper())
                                           for ii in range(site.NP)])
                        e = np.array([getattr(site.phase_tensors[ii],
                                              comp.upper() + '_error')
                                      for ii in range(site.NP)])
                    else:
                        toplot = np.array([getattr(site.phase_tensors[ii],
                                                   comp.lower())
                                           for ii in range(site.NP)])
                        e = np.array([getattr(site.phase_tensors[ii],
                                              comp.lower() + '_error', 0) # Default error of 0 until I implement errors for invariants
                                      for ii in range(site.NP)])
                    # Convert to degrees
                    if Type.lower() not in response_types and self.errors.lower() != 'none':
                        toplotErr = e
                    else:
                        toplotErr = e * 0
                    if self.pt_units.lower() == 'degrees':
                        toplot = np.rad2deg(np.arctan(toplot))
                        try:
                            toplotErr = np.rad2deg(np.arctan(toplotErr))
                        except AttributeError:  # If plotting PTs, toplotErr will be None
                            toplotErr = None
                elif 'bost' in comp.lower():
                    toplot, depth = utils.compute_bost1D(site, comp=comp)[:2]
                    toplot = np.log10(toplot)
                    toplotErr = None
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
                periods = site.periods
                if not self.plot_flagged_data:
                    if 'bost' in comp.lower():
                        depth, toplot, toplotErr = pop_flagged_data(depth, toplot, toplotErr, site, comp)
                    else:
                        periods, toplot, toplotErr = pop_flagged_data(periods, toplot, toplotErr, site, comp)
                    # depth = periods
                if 'i' in comp.lower():
                    if marker == 'o':
                        use_marker = 'v'
                        markersize = self.markersize * 1.25
                    else:
                        markersize = self.markersize
                        use_marker = marker
                else:
                    use_marker = marker
                    markersize = self.markersize
                    # marker = 'o'
                if 'bost' in comp.lower():
                    artist = ax.errorbar(np.log10(depth), toplot, xerr=None,
                                         yerr=None, marker=use_marker,
                                         linestyle=linestyle, color=self.get_designated_colour(comp, ii),
                                         mec=self.mec, markersize=markersize,
                                         mew=edgewidth, picker=3)
                                         # capsize=5)
                else:
                    # print(['Inside6', Type])
                    # if Type == 'smoothed_data':
                        # print(toplot)
                        # print([marker, linestyle])
                    artist = ax.errorbar(np.log10(periods), toplot, xerr=None,
                                         yerr=toplotErr, marker=use_marker,
                                         linestyle=linestyle, color=self.get_designated_colour(comp, ii),
                                         mec=self.mec, markersize=markersize,
                                         mew=edgewidth, picker=3)
                                         # capsize=5)
                if self.show_outliers and toplot.size != 0:
                    ma.append(max(toplot))
                    mi.append(min(toplot))
                else:
                    if ((self.toggles['raw_data'] and Type.lower() == 'raw_data') or \
                       (not self.toggles['raw_data'])) and (toplot != []):
                        # showdata = self.remove_outliers(site.periods, toplot)
                        showdata = utils.remove_outliers(toplot)
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
    COORD_SYSTEMS = ('local', 'utm', 'latlong', 'lambert')

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
        self.actors = {'annotation': [], 'locations': [], 'pseudosection': [],
                       'model': [], 'arrows': [], 'ellipses': []}
        self.site_marker = 'o'
        self.site_fill = True
        self.site_interior = 'k'
        self.arrow_colours = {'raw_data': {'R': 'b', 'I': 'r'},
                              'data': {'R': 'b', 'I': 'r'},
                              'response': {'R': 'g', 'I': 'c'}}
        self.linestyle = '-'
        self.site_exterior = {'generic': 'k', 'active': 'k'}
        self.markersize = 5
        self.edgewidth = 2
        self.image_opacity = 0.33
        self._coordinate_system = 'local'
        self.artist_ref = {'raw_data': [], 'data': [], 'response': []}
        self.annotate_sites = 'active'
        self.colourmap = 'turbo_r'
        self.has_nn = has_nn
        if self.has_nn:
            self.interpolant = 'natural'
        else:
            self.interpolant = 'nearest'
        self.rho_cax = [1, 5]
        self.phase_cax = [0, 90]
        self.phase_split_cax = [-40, 40]
        self.diff_cax = [-15, 15]
        self.tipper_cax = [0, 0.5]
        self.skew_cax = [-10, 10]
        self.model_cax = [1, 5]
        self.aniso_cax = [-2, 2]
        self.depth_cax = [0, 3.5]
        self.model = []
        self.padding_scale = 5
        self.plot_rms = False
        self.rms_plot_style = ['size']
        self.use_colourbar = True
        self.min_pt_ratio = 1 / 3
        self.pt_ratio_cutoff = 0
        self.pt_scale = 1
        self.ellipse_VE = 1
        self.ellipse_linewidth = 1
        self.pt_rotation_axis = 'x'
        self.induction_scale = 5
        self.induction_cutoff = 2
        self.induction_error_tol = 0.5
        self.rho_error_tol = 1
        self.phase_error_tol = 30
        self.include_outliers = True
        self.allowed_std = 3
        self.units = 'm'
        self.label_fontsize = 14
        self.mesh = False
        self.lut = 16
        self.linewidth = 0.005
        if figure is None:
            self.new_figure()
        else:
            self.window['figure'] = figure
        self.create_axes()

    @property
    def facecolour(self):
        if self.site_fill:
            return self.site_interior
        else:
            return 'none'

    @property
    def interpolant(self):
        return self._interpolant

    @interpolant.setter
    def interpolant(self, interpolant):
        if interpolant == 'natural':
            if not has_nn:
                interpolant = 'linear'
        self._interpolant = interpolant

    @property
    def cmap(self):
        return cm.get_cmap(self.colourmap, N=self.lut)

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
        coordinate_system = coordinate_system.lower()
        if coordinate_system in MapView.COORD_SYSTEMS and coordinate_system != self._coordinate_system:
            if self.verify_coordinate_system(coordinate_system):
                if coordinate_system in ('utm', 'lambert') and self._coordinate_system == 'local':
                    correction = utils.center_locs(np.fliplr(self.site_locations['all']))[1]
                else:
                    correction = np.array([0, 0])
                self._coordinate_system = coordinate_system
                generic_sites = list(set(self.site_names) - set(self.active_sites))
                self.site_locations['generic'] = self.get_locations(sites=generic_sites)
                self.site_locations['active'] = self.get_locations(sites=self.active_sites)
                self.site_locations['all'] = self.get_locations(sites=self.site_names)
                if self.model != []:
                    print('Correction factor is: {}'.format(correction))
                    
                    # print('Station center at: {}'.format(station_center))
                    station_center = np.array(utils.center_locs(np.fliplr(self.site_locations['all']))[1])
                    # if coordinate_system.lower() in ('utm', 'lambert'):
                        # self.model.project_model(system='local')
                    if coordinate_system == 'latlong':
                        print('Model to latlong not implemented')
                        return
                        # station_center = np.array(utils.center_locs((self.get_locations(sites=self.site_names,
                                                                                                 # coordinate_system='lambert',
                                                                                                 # inquire_only=True)))[1])
                    self.model.project_model(system=coordinate_system, origin=station_center - correction)
                    # elif coordinate_system.lower() == 'latlong':
                        # self.model.to_latlong()
                    # else:
                        # self.model.to_local()
                    # print('Model center shifted to: {}'.format(self.model.center))
                # if self.model != []:
                #     print('Correction factor is: {}'.format(correction))
                #     if coordinate_system.lower() == 'utm':
                #         station_center = np.array(utils.center_locs(np.fliplr(self.site_locations['all']))[1])
                #         print('Station center at: {}'.format(station_center))
                #         self.model.to_UTM(origin=station_center - correction)
                #     elif coordinate_system.lower() == 'lambert':
                #         station_center = np.array(utils.center_locs(np.fliplr(self.site_locations['all']))[1])
                #         print('Station center at: {}'.format(station_center))
                #         self.model.to_lambert(origin=station_center - correction)
                #     elif coordinate_system.lower() == 'latlong':
                #         self.model.to_latlong()
                #     else:
                #         self.model.to_local()
                #     print('Model center shifted to: {}'.format(self.model.center))
                # self.plot_locations()

    def verify_coordinate_system(self, coordinate_system):
        if coordinate_system.lower() == 'local':
            return True
        elif coordinate_system.lower() in ('utm', 'latlong', 'lambert'):
            if self.site_data['raw_data'].initialized:
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

    def get_locations(self, sites=None, coordinate_system=None, inquire_only=False):
        if not coordinate_system:
            coordinate_system = self.coordinate_system
        else:
            if not inquire_only:
                self.coordinate_system = coordinate_system
        if coordinate_system == 'local':
            azi = self.site_data['data'].azimuth
            check_azi = self.site_data['data'].check_azi()
            locs = self.site_data['data'].get_locs(site_list=sites, azi=azi)
        elif coordinate_system.lower() == 'utm':
            azi = self.site_data['raw_data'].azimuth
            check_azi = self.site_data['raw_data'].check_azi()
            locs = self.site_data['raw_data'].get_locs(sites=sites, mode='utm')
        elif coordinate_system.lower() == 'lambert':
            azi = self.site_data['raw_data'].azimuth
            check_azi = self.site_data['raw_data'].check_azi()
            locs = self.site_data['raw_data'].get_locs(sites=sites, mode='lambert')
        elif coordinate_system.lower() == 'latlong':
            azi = self.site_data['raw_data'].azimuth
            check_azi = self.site_data['raw_data'].check_azi()
            locs = self.site_data['raw_data'].get_locs(sites=sites, mode='latlong')
        if azi % 360 != check_azi % 360:
            print('Rotating')
            locs = utils.rotate_locs(locs, azi)
        return locs

    def set_axis_labels(self, xlabel=None, ylabel=None):
        if xlabel:
            self.window['axes'][0].set_xlabel(xlabel, fontsize=self.label_fontsize)
        else:
            if self.coordinate_system.lower() in ('utm', 'local', 'lambert'):
                self.window['axes'][0].set_xlabel('Easting (km)', fontsize=self.label_fontsize)
            else:
                self.window['axes'][0].set_xlabel(r'Longitude ($^{\circ}$)', fontsize=self.label_fontsize)
        if ylabel:
            self.window['axes'][0].set_ylabel(ylabel, fontsize=self.label_fontsize)
        else:
            if self.coordinate_system.lower() in ('utm', 'local', 'lambert'):
                self.window['axes'][0].set_ylabel('Northing (km)', fontsize=self.label_fontsize)
            else:
                self.window['axes'][0].set_ylabel(r'Latitude ($^{\circ}$)', fontsize=self.label_fontsize)

    def set_axis_limits(self, bounds=None, ax=None):
        if ax is None:
            ax = self.window['axes'][0]
        if bounds is None:
            min_x, max_x = (min(self.site_locations['all'][:, 1]),
                            max(self.site_locations['all'][:, 1]))
            min_y, max_y = (min(self.site_locations['all'][:, 0]),
                            max(self.site_locations['all'][:, 0]))
            x_pad = (max_x - min_x) / self.padding_scale
            y_pad = (max_y - min_y) / self.padding_scale
            ax.set_xlim([min_x - x_pad, max_x + x_pad])
            ax.set_ylim([min_y - y_pad, max_y + y_pad])
        else:
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])

    def interpolate(self, points, vals, n_interp):
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])
        x_pad = (max_x - min_x) / self.padding_scale
        min_x, max_x = (min_x - x_pad, max_x + x_pad)
        y_pad = (max_y - min_y) / self.padding_scale
        min_y, max_y = (min_y - y_pad, max_y + y_pad)
        X = np.linspace(min_x, max_x, n_interp)
        Y = np.linspace(min_y, max_y, n_interp)
        grid_x, grid_y = np.meshgrid(X, Y)
        if self.interpolant.lower() in ('linear', 'cubic', 'nearest'):
            gx, gy = np.mgrid[min_x:max_x:n_interp*1j, min_y:max_y:n_interp*1j]
            grid_vals = griddata(points[:,:2], vals, (gx, gy), method=self.interpolant)
        elif self.interpolant.lower() == 'natural' and has_nn:
            grid_ranges = [[min_x, max_x, n_interp * 1j],
                           [min_y, max_y, n_interp * 1j],
                           [0, 1, 1]]
            grid_vals = np.squeeze(nn.griddata(points, vals, grid_ranges))
        return grid_vals, grid_x, grid_y

    def plot_locations(self):
        # if not self.window['figure']:
        #     print('No figure to plot to...')
        #     return
        marker_size = {'generic': [self.markersize ** 2], 'active': [self.markersize ** 2]}
        facecolour = {'generic': 'None', 'active': 'None'}
        edgecolour = deepcopy(self.site_exterior)
        edgewidth = deepcopy(self.edgewidth)
        if self.plot_rms:
            marker = 'o'
            generic_rms = np.array([self.dataset.rms['Station'][site]['Total']
                                    for site in self.generic_sites])
            active_rms = np.array([self.dataset.rms['Station'][site]['Total']
                                   for site in self.active_sites])
            if 'size' in self.rms_plot_style:
                marker_size['generic'] = generic_rms
                marker_size['active'] = active_rms
                marker_size['generic'] = (utils.normalize(marker_size['generic'],
                                                          lower=1, upper=2, explicit_bounds=True) *
                                          self.markersize) ** 2
                marker_size['active'] = (utils.normalize(marker_size['active'],
                                                         lower=1, upper=2, explicit_bounds=True) *
                                         self.markersize) ** 2
            if 'colour' in self.rms_plot_style:
                facecolour['generic'] = generic_rms
                facecolour['active'] = active_rms
                edgecolour['generic'] = None
                edgecolour['active'] = None
        else:
            marker = self.site_marker
            facecolour['active'] = self.facecolour
            facecolour['generic'] = self.facecolour
            # edgewidth = 1
        if len(self.site_locations['generic']) > 0:
            try:
                self.window['axes'][0].scatter(self.site_locations['generic'][:, 1],
                                               self.site_locations['generic'][:, 0],
                                               marker=marker,
                                               s=marker_size['generic'],
                                               edgecolors=edgecolour['generic'],
                                               linewidths=edgewidth,
                                               # facecolors=facecolour,
                                               c=facecolour['generic'],
                                               zorder=9,
                                               cmap=self.cmap)
            except IndexError:
                pass

        if self.active_sites:
            self.window['axes'][0].scatter(self.site_locations['active'][:, 1],
                                           self.site_locations['active'][:, 0],
                                           marker=marker,
                                           s=marker_size['active'],
                                           edgecolors=edgecolour['active'],
                                           linewidths=edgewidth,
                                           # facecolors=facecolour,
                                           c=facecolour['active'],
                                           zorder=9,
                                           cmap=self.cmap)
        self.set_axis_labels()

    def plot_annotate(self):    
        if self.annotate_sites == 'active':
            for ii, (xx, yy) in enumerate(self.site_locations['active']):
                self.actors['annotation'].append(self.window['axes'][0].annotate(self.active_sites[ii], xy=(yy, xx)))
        elif self.annotate_sites == 'all':
            for ii, (xx, yy) in enumerate(self.site_locations['active']):
                    self.actors['annotation'].append(self.window['axes'][0].annotate(self.active_sites[ii], xy=(yy, xx)))
            for ii, (xx, yy) in enumerate(self.site_locations['generic']):
                self.actors['annotation'].append(self.window['axes'][0].annotate(self.generic_sites[ii], xy=(yy, xx)))
        self.set_axis_limits()

    @utils.enforce_input(data_type=list, normalize=bool, period_idx=int, arrow_type=list)
    def plot_induction_arrows(self, data_type='data', normalize=True, period_idx=1, arrow_type=['R']):
        # max_length = np.sqrt((np.max(self.site_locations['all'][:, 0]) -
        #                       np.min(self.site_locations['all'][:, 0])) ** 2 +
        #                      (np.max(self.site_locations['all'][:, 1]) -
        #                       np.min(self.site_locations['all'][:, 1])) ** 2) / 10
        x_max, x_min = (np.max(self.site_locations['all'][:, 0]), np.min(self.site_locations['all'][:, 0]))
        y_max, y_min = (np.max(self.site_locations['all'][:, 1]), np.min(self.site_locations['all'][:, 1]))
        max_length = np.max((x_max - x_min, y_max - y_min))
        num_keys = 0
        for R_or_L in arrow_type:
            for dType in data_type:
                X, Y = [], []
                colour = self.arrow_colours[dType][R_or_L]
                # if dType.lower() == 'data':
                #     colour = 'k'
                # elif dType.lower() == 'raw_data':
                #     colour = 'k'
                # elif dType.lower() == 'response':
                #     colour = 'r'
                idx = []
                for ii, site in enumerate(self.site_names):
                    site = self.site_data[dType].sites[site]
                    if np.all(np.array([site.used_error[comp][period_idx] for comp in site.TIPPER_COMPONENTS]) == site.REMOVE_FLAG):
                        continue
                    idx.append(ii)
                    if 'TZX' + R_or_L in site.components:
                        X.append(-site.data['TZX' + R_or_L][period_idx])
                    else:
                        X.append(0)
                    if 'TZY' + R_or_L in site.components:
                        Y.append(-site.data['TZY' + R_or_L][period_idx])
                    else:
                        Y.append(0)
                if idx:
                    arrows = np.transpose(np.array((X, Y)))
                    # arrows = utils.normalize_arrows(arrows)
                    lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                    preserved_lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                    largest_arrow = np.max(lengths)
                    lengths[lengths == 0] = 1
                    arrows[lengths > 1, 0] = 2 * arrows[lengths > 1, 0] / lengths[lengths > 1]
                    arrows[lengths > 1, 1] = 2 * arrows[lengths > 1, 1] / lengths[lengths > 1]
                    lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                    cutoff_idx = lengths > self.induction_cutoff

                    arrows[cutoff_idx, :] = 0
                    lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                    largest_arrow = np.max(lengths)

                    if normalize:
                        arrows = 0.5 * arrows / np.transpose(np.tile(lengths, [2, 1]))
                    arrows = arrows * self.induction_scale / (50) # * np.sqrt(lengths))
                    headwidth = max(3, 1)
                    # scale = self.induction_scale * max_length / 100
                    lengths = np.sqrt(arrows[:, 0] ** 2 + arrows[:, 1] ** 2)
                    largest_arrow = np.max(lengths)
                    width = 0.0025
                    quiv_handle = self.window['axes'][0].quiver(self.site_locations['all'][idx, 1],
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
                else:
                    preserved_lengths = 0
            # If no arrows are drawn
            if not idx:
                return
            # key_length_idx = np.argmin(np.abs(preserved_lengths - 0.5))
            key_length = 0.5 * self.induction_scale / 50
            horz_scale = (y_max - y_min) / 10
            vert_scale = (x_max - x_min) / 10
            label_pos_vert = x_min + vert_scale
            if R_or_L == 'R':
                key_label = 'Real'
            else:
                key_label = 'Imag'
            qk = self.window['axes'][0].quiverkey(quiv_handle,
                                                  self.window['axes'][0].get_xlim()[0] + horz_scale * 3.5,
                                                  self.window['axes'][0].get_ylim()[0] + vert_scale - num_keys * vert_scale / 2,
                                                  key_length, key_label + ', Length = 0.5',
                                                  labelpos='W',
                                                  coordinates='data')
            num_keys += 1

    def pt_fill_limits(self, fill_param, pt_type):
        if fill_param in ['Lambda']:
            lower, upper = (0, 0.5)
        elif fill_param == 'beta':
            lower, upper = self.skew_cax
        elif fill_param == 'absbeta':
            lower, upper = [0, self.skew_cax[1]]
        elif fill_param in ['alpha', 'azimuth']:
            lower, upper = (-90, 90)
            # lower, upper = (0, 180)
        elif fill_param in ('delta'):
            lower, upper = (0, 100)
        elif fill_param in ('phi_split', 'phi_split_z', 'phi_split_pt'):
            if fill_param == 'phi_split_pt':
                lower, upper = [0, self.phase_split_cax[1]]
            else:
                lower, upper = self.phase_split_cax
        elif fill_param.lower() == 'dimensionality':
            lower, upper = (1, 3)
        else:
            if pt_type.lower() == 'phi':
                if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3', 'phi_split', 'phi_split_z', 'phi_split_pt']:
                    lower, upper = self.phase_cax
            elif pt_type.lower() == 'phi_a':
                if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3', 'phi_split', 'phi_split_z', 'phi_split_pt']:
                    lower, upper = [-1*self.phase_cax[1], self.phase_cax[1]]
            elif pt_type.lower() == 'ua':
                if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3', 'phi_split', 'phi_split_z', 'phi_split_pt']:
                    lower, upper = self.rho_cax
            elif pt_type.lower() == 'va':
                if fill_param in ['phi_max', 'phi_min', 'det_phi', 'phi_1', 'phi_2', 'phi_3', 'phi_split', 'phi_split_z', 'phi_split_pt']:
                    lower, upper = [-1 * self.rho_cax[1], self.rho_cax[1]]
        return lower, upper

    def generate_rectangle(self, width, height, angle):
        t = np.arange(0, 2 * np.pi + np.pi / 30, np.pi / 30)
        x = width * (abs(np.cos(t)) * np.cos(t) + abs(np.sin(t)) * np.sin(t))
        y = height * (abs(np.cos(t)) * np.cos(t) - abs(np.sin(t)) * np.sin(t))
        R = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
        x, y = np.matmul(R, np.array((x, y)))
        return x, y

    def generate_ellipse(self, phi):
            jx = np.cos(np.arange(0, 2 * np.pi + np.pi / 30, np.pi / 30))
            jy = np.sin(np.arange(0, 2 * np.pi + np.pi / 30, np.pi / 30))
            phi_x = phi[0, 0] * jx + phi[0, 1] * jy
            phi_y = phi[1, 0] * jx + phi[1, 1] * jy
            # phi_x = jx
            # phi_y = jy
            return phi_x, phi_y

    def get_tensor_params(self, pt_type, fill_param, site=None, tensor=None, period_idx=None):
        if pt_type == 'phi':
            if not tensor:
                tensor = site.phase_tensors[period_idx]
            phi = tensor.phi
            phi_max = tensor.phi_max
            phi_min = tensor.phi_min
            azimuth = tensor.azimuth
            fill_val = getattr(tensor, fill_param)
        elif pt_type == 'phi_a':
            if not tensor:
                tensor = site.CART[period_idx]
            phi = tensor.phi
            phi_max = tensor.phi_max
            phi_min = tensor.phi_min
            azimuth = tensor.azimuth
            fill_val = getattr(tensor, fill_param)
        else:
            if not tensor:
                tensor = site.CART[period_idx]
            phi = getattr(tensor, pt_type)
            phi_max = getattr(tensor, '_'.join([pt_type, 'phi_max']))
            phi_min = getattr(tensor, '_'.join([pt_type, 'phi_min']))
            azimuth = getattr(tensor, '_'.join([pt_type, 'azimuth']))
            fill_val = getattr(tensor, '_'.join([pt_type, fill_param]))
            if fill_param in ('phi_1', 'phi_2', 'phi_3', 'phi_min', 'phi_max', 'det_phi'):
                if pt_type.lower() == 'ua':
                    fill_val = np.log10(abs(fill_val))
                elif pt_type.lower() == 'va':
                    fill_val = np.sign(fill_val)*np.log10(abs(fill_val))
        return tensor, phi, phi_max, phi_min, azimuth, fill_val

    def resize_ellipse(self, azimuth, phi_max):
        R = np.array(((np.cos(azimuth), -np.sin(azimuth)),
                       (np.sin(azimuth), np.cos(azimuth))))
        max_element = np.abs(phi_max)
        minimal_phi = np.array(((max_element, 0), (0, max_element * self.min_pt_ratio)))
        phi_x, phi_y = self.generate_ellipse(minimal_phi)
        for kk in range(len(phi_x)):
            phi_new = np.matmul(R, np.array((phi_x[kk], phi_y[kk])))
            phi_x[kk], phi_y[kk] = phi_new
        return phi_x, phi_y

    @utils.enforce_input(data_type=list, normalize=bool, fill_param=str, period_idx=int, pt_type=str, bostick_depth=float)
    def plot_phase_tensor(self, data_type='data', normalize=True, fill_param='Beta', period_idx=1, pt_type=None, bostick_depth=0):
        ellipses = []
        fill_vals = []
        if not pt_type:
            pt_type = 'phi'
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
                if bostick_depth:
                    depths = utils.compute_bost1D(site=site, method='phase', comp='aav', filter_width=1)[1]
                    period_idx = np.argmin(abs(bostick_depth - depths))
                tensor, phi, phi_max, phi_min, azimuth, fill_val = self.get_tensor_params(pt_type,
                                                                                          fill_param,
                                                                                          site=site,
                                                                                          period_idx=period_idx)

                # if ((phase_tensor.rhoxy_error / phase_tensor.rhoxy < self.rho_error_tol) and
                #     (phase_tensor.rhoyx_error / phase_tensor.rhoyx < self.rho_error_tol) and
                #     (phase_tensor.phasexy_error < self.phase_error_tol) and
                #     (phase_tensor.phaseyx_error < self.phase_error_tol)):
                if True:
                    cont = 1
                    phi_x, phi_y = self.generate_ellipse(phi)
                    radii = np.sqrt(phi_x ** 2 + phi_y ** 2)
                    if np.min(radii) < np.max(radii) * self.pt_ratio_cutoff:
                        cont = 0

                    if np.all(np.array([site.used_error[comp][period_idx] for comp in site.components]) == site.REMOVE_FLAG):
                        cont = 0
                    if np.min(radii) < np.max(radii) * self.min_pt_ratio:
                        phi_x, phi_y = self.resize_ellipse(azimuth, phi_max)

                    norm_x, norm_y = (phi_x, phi_y)
                    # good_idx.append(ii)
                    # print('Site: {} Phase error: {}, Rho error {}'.format(site_name, phaseyx_error, rhoyx_error))
                else:
                    cont = 0
                    # print('Site: {} Phase error: {}, Rho error {}'.format(site_name, phaseyx_error, rhoyx_error))
                    phi_x, phi_y = self.generate_ellipse(phi)
                    norm_x, norm_y = (phi_x, phi_y)
            else:
                if True:  # Buffer for error tolerances at some point...
                    tensor1 = self.get_tensor_params(pt_type,
                                                     fill_param,
                                                     site=self.site_data[data_type[0]].sites[site_name],
                                                     period_idx=period_idx)[0]
                    tensor2 = self.get_tensor_params(pt_type,
                                                     fill_param,
                                                     site=self.site_data[data_type[1]].sites[site_name],
                                                     period_idx=period_idx)[0]

                    residual_tensor = tensor1 - tensor2
                    tensor, phi, phi_max, phi_min, azimuth, fill_val  = self.get_tensor_params(pt_type,
                                                                                               fill_param, 
                                                                                               tensor=residual_tensor)
                    phi_x, phi_y = self.generate_ellipse(phi)
                    radii = np.sqrt(phi_x ** 2 + phi_y ** 2)
                    if np.min(radii) < np.max(radii) * self.pt_ratio_cutoff:
                        cont = 0
                    else:
                        cont = 1
                    if np.min(radii) < np.max(radii) * self.min_pt_ratio:
                        phi_x, phi_y = self.resize_ellipse(phi_max, azimuth)
                                       
                    # norm_x, norm_y = generate_ellipse(self.site_data[data_type[0]].sites[site_name].phase_tensors[period_idx].phi)
                    
            if cont:
                good_idx.append(ii)
                X, Y = X_all[ii], Y_all[ii]
                phi_x, phi_y = (1000 * phi_x / np.abs(phi_max),
                                1000 * phi_y / np.abs(phi_max))
                radius = np.max(np.sqrt(phi_x ** 2 + phi_y ** 2))
                # if radius > 1000:
                phi_x, phi_y = [(self.pt_scale * scale / (radius * 100)) * x for x in (phi_x, phi_y)]
                ellipses.append([Y - phi_x, X - phi_y])
                # fill_vals.append(getattr(phase_tensor, fill_param))
                fill_vals.append(fill_val)
        fill_vals = np.array(fill_vals)
        lower, upper = self.pt_fill_limits(fill_param, pt_type)
        # Alpha, beta, and therefore azimuth are already arctan'ed in data_structures
        if fill_param not in ('delta', 'Lambda', 'alpha', 'azimuth', 'beta', 'phi_split', 'phi_split_z', 'phi_split_pt', 'dimensionality') and pt_type in ('phi', 'phi_a'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        if fill_param in ['alpha', 'azimuth', 'beta']:
            fill_vals = np.rad2deg(fill_vals)
                # fill_vals[fill_vals < 0] = 180 + fill_vals[fill_vals < 0]
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        for ii, ellipse in enumerate(ellipses):
            self.window['axes'][0].fill(ellipse[0], ellipse[1],
                                        color=self.cmap(norm_vals[ii]),
                                        zorder=3,
                                        edgecolor='k',
                                        linewidth=0)
            if self.ellipse_linewidth:
                self.window['axes'][0].plot(ellipse[0], ellipse[1],
                                            'k-', linewidth=self.ellipse_linewidth, zorder=3)
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

    @utils.enforce_input(data_type=list, normalize=bool, fill_param=str, period_idx=int, pt_type=str)
    def plot_phase_bar(self, data_type='data', normalize=True, fill_param='Beta', period_idx=1, pt_type=None):
        rectangles = []
        fill_vals = []
        if not pt_type:
            pt_type = 'phi'
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
            if np.all(np.array([site.used_error[comp][period_idx] for comp in site.components]) == site.REMOVE_FLAG):
                continue
            tensor, phi, phi_max, phi_min, azimuth, fill_val = self.get_tensor_params(pt_type,
                                                                                      fill_param,
                                                                                      site=site,
                                                                                      period_idx=period_idx)
            xy = [Y_all[ii], X_all[ii]]
            # width = np.abs(phi_max / 5)
            if np.abs(phi_min / phi_max) < self.pt_ratio_cutoff:
                continue
            if np.abs(phi_min / phi_max) < self.min_pt_ratio:
                height = 2 * self.min_pt_ratio
            else:
                height = np.abs(2 * phi_min / phi_max)
            width = 0.2
            # height = 1
            width, height = [(self.pt_scale * scale) * x / 100 for x in (width, height)]
            xy[0] = xy[0] + height * (np.sin(azimuth)) / 2 - width * (np.cos(azimuth)) / 2
            xy[1] = xy[1] - height * (np.cos(azimuth)) / 2 - width * (np.sin(azimuth)) / 2
            rectangles.append((xy, width, height, np.rad2deg(azimuth)))
            #     ellipses.append([Y - phi_x, X - phi_y])
            fill_vals.append(fill_val)
        fill_vals = np.array(fill_vals)
        lower, upper = self.pt_fill_limits(fill_param, pt_type)
        # Alpha, beta, and therefore azimuth are already arctan'ed in data_structures
        if fill_param not in ('delta', 'Lambda', 'alpha', 'azimuth', 'beta', 'phi_split', 'phi_split_z', 'phi_split_pt', 'dimensionality') and pt_type in ('phi', 'phi_a'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        if fill_param in ['alpha', 'azimuth', 'beta']:
            fill_vals = np.rad2deg(fill_vals)
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
                                     color=self.cmap(norm_vals[ii]),
                                     zorder=4)
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
                data_label = 'Resistivity'
                use_log = True
                if '-' in fill_param:
                    val = []
                    for site in data.site_names:
                        val.append(utils.compute_rho(data.sites[site], calc_comp='xy', errtype='none')[0][period_idx] -
                                   utils.compute_rho(data.sites[site], calc_comp='yx', errtype='none')[0][period_idx])
                        val[-1] = np.sign(val[-1])*np.log10(np.abs(val[-1]))
                    vals.append(val)
                    # vals.append([np.log10(utils.compute_rho(site=data.sites[site], calc_comp='xy', errtype='none')[0][period_idx] - 
                    #                       utils.compute_rho(site=data.sites[site], calc_comp='yx', errtype='none')[0][period_idx]) for site in data.site_names])
                else:
                    vals.append([np.log10(utils.compute_rho(site=data.sites[site],
                                                            calc_comp=fill_param,
                                                            errtype='none')[0][period_idx]) for site in data.site_names])
                # vals.append(temp_vals)
            elif 'pha' in fill_param.lower():
                data_label = 'Phase'
                use_log = False
                if '-' in fill_param:
                    vals.append([utils.compute_phase(data.sites[site], calc_comp='xy', errtype='none', wrap=1)[0][period_idx] -
                                 utils.compute_phase(data.sites[site], calc_comp='yx', errtype='none', wrap=1)[0][period_idx] for site in data.site_names])
                else:
                    vals.append([utils.compute_phase(site=data.sites[site],
                                                     calc_comp=fill_param,
                                                     errtype='none',
                                                     wrap=1)[0][period_idx] for site in data.site_names])
            elif 'tip' in fill_param.lower():
                data_label = 'Tipper Amplitude'
                use_log = False
                vals.append([np.sqrt((data.sites[site].data['TZXR'][period_idx] ** 2 +
                                     data.sites[site].data['TZYR'][period_idx] ** 2)) for site in data.site_names])
            elif 'absbeta' in fill_param.lower():
                data_label = 'Skew'
                use_log = False
                vals.append([abs(np.rad2deg(data.sites[site].phase_tensors[period_idx].beta)) for site in data.site_names])
            elif 'beta' in fill_param.lower():
                data_label = 'Skew'
                use_log = False
                vals.append([np.rad2deg(data.sites[site].phase_tensors[period_idx].beta) for site in data.site_names])
            elif 'azimuth' in fill_param.lower():
                data_label = 'Azimuth'
                use_log = False
                vals.append([np.rad2deg(data.sites[site].phase_tensors[period_idx].azimuth) for site in data.site_names])
            elif 'bost' in fill_param.lower():
                data_label = 'Depth'
                use_log = True
                # vals.append([np.log10(utils.compute_bost1D(data.sites[site], method='phase', comp=fill_param, filter_width=1)[1][period_idx]) for site in data.site_names])
                if '-' in fill_param:
                    val = []
                    for site in data.site_names:
                        val.append(utils.compute_bost1D(data.sites[site], method='bostick', comp='xy', filter_width=1)[1][period_idx] -
                                   utils.compute_bost1D(data.sites[site], method='bostick', comp='yx', filter_width=1)[1][period_idx])
                        val[-1] = np.sign(val[-1])*np.log10(np.abs(val[-1]))
                    vals.append(val)

                    # vals.append([(utils.compute_bost1D(data.sites[site], method='bostick', comp='xy', filter_width=1)[1][period_idx] -
                                  # utils.compute_bost1D(data.sites[site], method='bostick', comp='yx', filter_width=1)[1][period_idx]) for site in data.site_names])
                else:
                    vals.append([np.log10(utils.compute_bost1D(data.sites[site], method='bostick', comp=fill_param, filter_width=1)[1][period_idx]) for site in data.site_names])
            elif 'pt_split' in fill_param.lower():
                data_label = 'Phase Split'
                use_log = False
                vals.append([data.sites[site].phase_tensors[period_idx].phi_split_pt for site in data.site_names])
            elif 'phi_min' in fill_param.lower():
                data_label = 'Phase'
                use_log = False
                vals.append([np.rad2deg(np.arctan(data.sites[site].phase_tensors[period_idx].phi_min)) for site in data.site_names])
            elif 'phi_max' in fill_param.lower():
                data_label = 'Phase'
                use_log = False
                vals.append([np.rad2deg(np.arctan(data.sites[site].phase_tensors[period_idx].phi_max)) for site in data.site_names])
        if len(vals) == 1:
            vals = np.array(vals[0])
            diff = False
        else:
            diff = True
            #  Should be response - data so a larger response gives a +ve percent difference
            if 'rho' in fill_param.lower() or 'bost' in fill_param.lower():
                # vals = 100 * (np.array(vals[1]) - np.array(vals[0])) / np.array(vals[0])
                vals = (np.array(vals[1])) - (np.array(vals[0]))
                use_log = False
            # elif 'pha' in fill_param.lower() or 'tip' in fill_param.lower():
            else:
                vals = (np.array(vals[1]) - np.array(vals[0]))  # / np.array(vals[0])
        if not self.include_outliers:
            val_mean = np.mean(vals)
            val_std = np.std(vals)
            good_idx = ((vals < (val_mean + self.allowed_std * val_std)) &
                        (vals > (val_mean - self.allowed_std * val_std)))
        elif 'tip' in fill_param.lower():
            good_idx = vals < self.induction_cutoff
        else:
            good_idx = abs(vals) < 1e10
        loc_x, loc_y = self.site_locations['all'][good_idx, 1], self.site_locations['all'][good_idx, 0]
        loc_z = np.zeros(loc_x.shape)
        points = np.transpose(np.array((loc_x, loc_y, loc_z)))
        min_x, max_x = (min(loc_x), max(loc_x))
        x_pad = (max_x - min_x) / self.padding_scale
        min_x, max_x = (min_x - x_pad, max_x + x_pad)
        min_y, max_y = (min(loc_y), max(loc_y))
        y_pad = (max_y - min_y) / self.padding_scale
        min_y, max_y = (min_y - y_pad, max_y + y_pad)
        X = np.linspace(min_x, max_x, n_interp)
        Y = np.linspace(min_y, max_y, n_interp)
        grid_vals, grid_x, grid_y = self.interpolate(points, vals[good_idx], n_interp)
        if diff and 'rho' in fill_param.lower():
            cax = self.diff_cax
            fill_param = 'Log10 Difference'
        elif (diff and 'pha' in fill_param.lower() or fill_param.lower() == ('phaxy-yx')) or \
             (diff and 'pt_split' in fill_param.lower()) or (diff and 'phi' in fill_param.lower()):
            cax = self.diff_cax
            fill_param = r'Difference ($^{\circ}$)'
        # Any other difference just use a generic label
        elif diff: # and 'tip' in fill_param.lower(): 
            cax = self.diff_cax
            fill_param = r'Difference'
        elif 'rho' in fill_param.lower():
            cax = self.rho_cax
        elif 'pha' in fill_param.lower() or 'phi' in fill_param.lower():
            cax = self.phase_cax
        elif 'tip' in fill_param.lower():
            cax = self.tipper_cax
        elif 'absbeta' in fill_param.lower():
            cax = [0, self.skew_cax[1]]
        elif 'beta' in fill_param.lower():
            cax = self.skew_cax
        elif 'azimuth' in fill_param.lower():
            cax = [-90, 90]
        elif 'bost' in fill_param.lower():
            cax = self.depth_cax
        elif 'pt_split' in fill_param.lower():
            cax = [0, self.phase_split_cax[1]]
        
        im = self.window['axes'][0].pcolor(grid_x, grid_y, grid_vals.T,
                                           cmap=self.cmap,
                                           vmin=cax[0], vmax=cax[1],
                                           zorder=0)
        if self.window['colorbar'] and self.use_colourbar:
            self.window['colorbar'].remove()
            self.window['colorbar'] = self.window['figure'].colorbar(mappable=im)
            self.window['colorbar'].set_label(fill_param,
                                              rotation=270,
                                              labelpad=20,
                                              fontsize=18)
        self.set_axis_limits()
        self.window['axes'][0].format_coord = format_model_coords(im,
                                              X=X, Y=Y,
                                              x_label='Easting', y_label='Northing',
                                              data_label=data_label,
                                              use_log=use_log)

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
            label = r'$\phi_{max}$'
        elif param.lower() == 'phi_min':
            label = r'$\phi_{min}$'
        else:
            label = ''
        return label

    def plot_plan_view(self, ax=None, z_slice=0, rho_axis='rho_x'):
        rho_axis = rho_axis.lower()
        if not ax:
            ax = self.window['axes'][0]
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        if '/' in rho_axis:
            rho_axis = rho_axis.split('/')
            v1 = getattr(self.model, rho_axis[0].strip())
            v2 = getattr(self.model, rho_axis[1].strip())
            vals = v1 / v2
            cax = self.aniso_cax
        else:
            vals = getattr(self.model, rho_axis, 'vals')
            cax = self.model_cax
        # debug_print(vals, 'test.txt')
        X = np.array(self.model.dy)
        Y = np.array(self.model.dx)
        im = ax.pcolormesh(X,
                           Y,
                           np.log10(vals[:, :, z_slice]),
                           cmap=self.cmap,
                           vmin=cax[0], vmax=cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        self.set_axis_labels()
        # ax.set_ylabel('Northing (km)')
        # ax.set_xlabel('Easting (km)')
        ax.format_coord = format_model_coords(im,
                                              X=X, Y=Y,
                                              x_label='Easting', y_label='Northing')

    def plot_x_slice(self, ax=None, x_slice=0, rho_axis='rho_x'):
        rho_axis = rho_axis.lower()
        if not ax:
            ax = self.window['axes'][0]
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        if '/' in rho_axis:
            rho_axis = rho_axis.split('/')
            v1 = getattr(self.model, rho_axis[0].strip())
            v2 = getattr(self.model, rho_axis[1].strip())
            vals = v1 / v2
            cax = self.aniso_cax
        else:
            vals = getattr(self.model, rho_axis, 'vals')
            cax = self.model_cax
        X = np.array(self.model.dy)
        Y = np.array(self.model.dz)
        im = ax.pcolormesh(X,
                           Y,
                           np.squeeze(np.log10(vals[x_slice, :, :])).T,
                           cmap=self.cmap,
                           vmin=cax[0], vmax=cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        self.set_axis_labels()
        # ax.set_xlabel('Easting (km)')
        # ax.set_ylabel('Depth (km)')
        ax.format_coord = format_model_coords(im,
                                              X=X, Y=Y,
                                              x_label='Easting', y_label='Depth')

    def plot_y_slice(self, ax=None, y_slice=0, orientation='xz', rho_axis='rho_x'):
        rho_axis = rho_axis.lower()
        if not ax:
            ax = self.window['axes'][0]
        if self.mesh:
            edgecolor = 'k'
        else:
            edgecolor = None
        if '/' in rho_axis:
            rho_axis = rho_axis.split('/')
            v1 = getattr(self.model, rho_axis[0].strip())
            v2 = getattr(self.model, rho_axis[1].strip())
            vals = v1 / v2
            cax = self.aniso_cax
        else:
            vals = getattr(self.model, rho_axis, 'vals')
            cax = self.model_cax
        if orientation.lower() == 'xz':
            X = np.array(self.model.dx)
            Y = np.array(self.model.dz)
            to_plot = np.squeeze(np.log10(vals[:, y_slice, :])).T
            x_label = 'Northing (km)'
            y_label = 'Depth (km)'
        else:
            Y = np.array(self.model.dx)
            X = np.array(self.model.dz)
            to_plot = np.squeeze(np.log10(vals[:, y_slice, :]))
            y_label = 'Northing (km)'
            x_label = 'Depth (km)'
        im = ax.pcolormesh(X, Y, to_plot,
                           cmap=self.cmap,
                           vmin=cax[0], vmax=cax[1],
                           zorder=0,
                           edgecolor=edgecolor,
                           linewidth=self.linewidth)
        self.set_axis_labels(xlabel=x_label, ylabel=y_label)
        # ax.set_xlabel(x_label)
        # ax.set_ylabel(y_label)
        ax.format_coord = format_model_coords(im, X=X, Y=Y,
                                              x_label=x_label.split()[0],
                                              y_label=y_label.split()[0])

    def plot_image(self, image, extents, ax=None):
        if not ax:
            ax = self.window['axes'][0]
        ax.imshow(image, extent=extents, alpha=self.image_opacity, zorder=2)

    @utils.enforce_input(sites=list, data_type=str, fill_param=str, periods=tuple, pt_type=str, x_axis=str, annotate_sites=bool)
    def tensor_ellipse_pseudosection(self, sites=None, data_type='data', fill_param='beta', periods=(), pt_type=None, x_axis='linear', annotate_sites=False):
        # Here periods is defined by the tuple (lower, upper, skip). Done this way to ensure we can treat both raw and used data
        ellipses = []
        fill_vals = []
        if not pt_type:
            pt_type = 'phi'
        if fill_param != 'Lambda':
            fill_param = fill_param.lower()

        if x_axis.lower() == 'x':
            X = np.array([self.site_data[data_type].sites[site_name].locations['X'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = 'Northing ({})'.format(self.site_data[data_type].spatial_units)
        elif x_axis.lower() == 'y':
            X = np.array([self.site_data[data_type].sites[site_name].locations['Y'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = 'Easting ({})'.format(self.site_data[data_type].spatial_units)
        elif x_axis.lower() in ('lat', 'latitude'):
            X = np.array([self.site_data[data_type].sites[site_name].locations['Lat'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = r'Latitude ($^{\circ}$)'
        elif x_axis.lower() in ('lon', 'long', 'longitude'):
            X = np.array([self.site_data[data_type].sites[site_name].locations['Long'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = r'Longitude ($^{\circ}$)'
        elif x_axis.lower() == 'linear':
            # If linear distance is used, assume the user has the sites in the desired order
            x = np.array([self.site_data[data_type].sites[site_name].locations['X'] for site_name in sites])
            y = np.array([self.site_data[data_type].sites[site_name].locations['Y'] for site_name in sites])
            X = utils.linear_distance(x, y)
            label = 'Distance ({})'.format(self.site_data[data_type].spatial_units)
        # scale = np.sqrt((np.max(X) - np.min(X)) ** 2 +
                        # (np.max(np.log10(periods[0])) - np.log10(np.min(periods[1]))) ** 2)
        x_scale = (np.max(X) - np.min(X))
        y_scale = (np.log10(periods[1]) - np.log10(periods[0]))
        # print([x_scale, y_scale])
        X_all, Y_all = [], []
        for ii, site_name in enumerate(sites):
            site = self.site_data[data_type].sites[site_name]
            for ip, period in enumerate(site.periods):
                if (period >= periods[0] and period <= periods[1]) and (ip % (periods[2] + 1)) == 0:
                    X_all.append(X[ii])
                    Y_all.append(np.log10(period))
                    tensor, phi, phi_max, phi_min, azimuth, fill_val = self.get_tensor_params(pt_type,
                                                                                              fill_param,
                                                                                              site=site,
                                                                                              period_idx=ip)
                    phi_x, phi_y = self.generate_ellipse(phi)
                    radii = np.sqrt(phi_x ** 2 + phi_y ** 2)
                    if np.min(radii) < np.max(radii) * self.pt_ratio_cutoff:
                        X_all.pop()
                        Y_all.pop()
                        continue
                    if np.min(radii) < np.max(radii) * self.min_pt_ratio:
                        phi_x, phi_y = self.resize_ellipse(azimuth, phi_max)
                    current_x, current_y = X_all[-1], Y_all[-1]
                    phi_x, phi_y = (1000 * phi_x / np.abs(phi_max),
                                    1000 * phi_y / np.abs(phi_max))
                    radius = np.max(np.sqrt(phi_x ** 2 + phi_y ** 2))
                    # if radius > 1000:
                    phi_x = [self.ellipse_VE * (self.pt_scale * x_scale / (radius * 100)) * x for x in phi_x]
                    phi_y = [(self.pt_scale * y_scale / (radius * 100)) * y for y in phi_y]
                    ellipses.append([current_x - phi_x, current_y - phi_y])
                    fill_vals.append(fill_val)
        fill_vals = np.array(fill_vals)
        lower, upper = self.pt_fill_limits(fill_param, pt_type)
        # Alpha, beta, and therefore azimuth are already arctan'ed in data_structures
        if fill_param not in ('delta', 'Lambda', 'alpha', 'azimuth', 'beta', 'phi_split', 'phi_split_z', 'phi_split_pt', 'dimensionality') and pt_type in ('phi', 'phi_a'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        if fill_param in ['alpha', 'azimuth', 'beta']:
            fill_vals = np.rad2deg(fill_vals)
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        for ii, ellipse in enumerate(ellipses):
            self.window['axes'][0].fill(ellipse[0], ellipse[1],
                                        color=self.cmap(norm_vals[ii]),
                                        zorder=3,
                                        edgecolor='k',
                                        linewidth=0)
            self.window['axes'][0].plot(ellipse[0], ellipse[1],
                                        'k-', linewidth=self.ellipse_linewidth, zorder=3)
        fake_vals = np.linspace(lower, upper, len(fill_vals))
        self.fake_im = self.window['axes'][0].scatter(X_all,
                                                      Y_all,
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
        x_diff = abs(max(X_all) - min(X_all))
        y_diff = abs(max(Y_all) - min(Y_all))
        x_min, x_max = min(X_all), max(X_all)
        y_min, y_max = min(Y_all), max(Y_all)
        xlim = [x_min - x_diff/10, x_max + x_diff/10]
        ylim = [y_min - y_diff/10, y_max + y_diff/10]
        # print([x_min, x_max, x_diff])
        # print([y_min, y_max, y_diff])        
        aspect = self.ellipse_VE * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        self.set_axis_limits(bounds=xlim + ylim)
        self.window['axes'][0].set_aspect(aspect)
        self.window['axes'][0].invert_yaxis()

    @utils.enforce_input(sites=list, data_type=str, fill_param=str, periods=tuple, pt_type=str, x_axis=str)
    def tensor_bar_pseudosection(self, sites=None, data_type='data', fill_param='beta', periods=(), pt_type=None, x_axis='linear'):
        # Here periods is defined by the tuple (lower, upper, skip). Done this way to ensure we can treat both raw and used data
        rectangles = []
        fill_vals = []
        if not pt_type:
            pt_type = 'phi'
        if fill_param != 'Lambda':
            fill_param = fill_param.lower()

        if x_axis.lower() == 'x':
            X = np.array([self.site_data[data_type].sites[site_name].locations['X'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = 'Northing ({})'.format(self.site_data[data_type].spatial_units)
        elif x_axis.lower() == 'y':
            X = np.array([self.site_data[data_type].sites[site_name].locations['Y'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = 'Easting ({})'.format(self.site_data[data_type].spatial_units)
        elif x_axis.lower() in ('lat', 'latitude'):
            X = np.array([self.site_data[data_type].sites[site_name].locations['Lat'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = r'Latitude ($^{\circ}$)'
        elif x_axis.lower() in ('lon', 'long', 'longitude'):
            X = np.array([self.site_data[data_type].sites[site_name].locations['Long'] for site_name in sites])
            idx = np.argsort(X)
            X = X[idx]
            sites = [sites[c] for c in idx]
            label = r'Longitude ($^{\circ}$)'
        elif x_axis.lower() == 'linear':
            # If linear distance is used, assume the user has the sites in the desired order
            x = np.array([self.site_data[data_type].sites[site_name].locations['X'] for site_name in sites])
            y = np.array([self.site_data[data_type].sites[site_name].locations['Y'] for site_name in sites])
            X = utils.linear_distance(x, y)
            label = 'Distance ({})'.format(self.site_data[data_type].spatial_units)
        scale = np.sqrt((np.max(X) - np.min(X)) ** 2 +
                        (np.max(np.log10(periods[0])) - np.log10(np.min(periods[1]))) ** 2)
        x_scale = np.sqrt((np.max(X) - np.min(X)) ** 2)
        y_scale = np.sqrt((np.log10(periods[1]) - np.log10(periods[0])) ** 2)
        X_all, Y_all = [], []
        for ii, site_name in enumerate(sites):
            site = self.site_data[data_type].sites[site_name]
            for ip, period in enumerate(site.periods):
                if (period >= periods[0] and period <= periods[1]) and (ip % (periods[2] + 1)) == 0:
                    X_all.append(X[ii])
                    Y_all.append(np.log10(period))
                    tensor, phi, phi_max, phi_min, azimuth, fill_val = self.get_tensor_params(pt_type,
                                                                                              fill_param,
                                                                                              site=site,
                                                                                              period_idx=ip)
                    xy = [X_all[-1], Y_all[-1]]
                    # width = np.abs(phi_max / 5)
                    if np.abs(phi_min / phi_max) < self.pt_ratio_cutoff:
                        X_all.pop()
                        Y_all.pop()
                        continue
                    if np.abs(phi_min / phi_max) < self.min_pt_ratio:
                        height = 2 * self.min_pt_ratio
                    else:
                        height = np.abs(2 * phi_min / phi_max)
                    width = 0.2
                    height = height / 2
                    x, y = self.generate_rectangle(width, height, azimuth)
                    # # x *= (self.pt_scale * x_scale / (100))
                    # # y *= (self.pt_scale * y_scale / (100))
                    # phi_x = [self.ellipse_VE * (self.pt_scale * x_scale / (radius * 100)) * x for x in phi_x]
                    x = self.ellipse_VE * (self.pt_scale * x_scale / (100)) * x
                    y = (self.pt_scale * y_scale / (100)) * y
                    rectangles.append([xy[0] - x, xy[1] - y])
                    fill_vals.append(fill_val)
        fill_vals = np.array(fill_vals)
        lower, upper = self.pt_fill_limits(fill_param, pt_type)
        # Alpha, beta, and therefore azimuth are already arctan'ed in data_structures
        if fill_param not in ('delta', 'Lambda', 'alpha', 'azimuth', 'beta', 'phi_split', 'phi_split_z', 'phi_split_pt', 'dimensionality') and pt_type in ('phi', 'phi_a'):
            fill_vals = np.rad2deg(np.arctan(fill_vals))
        if fill_param in ['alpha', 'azimuth', 'beta']:
            fill_vals = np.rad2deg(fill_vals)
        fill_vals[fill_vals > upper] = upper
        fill_vals[fill_vals < lower] = lower
        norm_vals = utils.normalize_range(fill_vals,
                                          lower_range=lower,
                                          upper_range=upper,
                                          lower_norm=0,
                                          upper_norm=1)
        for ii, rectangle in enumerate(rectangles):
            self.window['axes'][0].fill(rectangle[0], rectangle[1],
                                        color=self.cmap(norm_vals[ii]),
                                        zorder=4,
                                        edgecolor=None)
        self.window['axes'][0].invert_yaxis()
        x_diff = abs(max(X_all) - min(X_all))
        y_diff = abs(max(Y_all) - min(Y_all))
        x_min, x_max = min(X_all), max(X_all)
        y_min, y_max = min(Y_all), max(Y_all)
        # print([x_min, x_max, x_diff])
        # print([y_min, y_max, y_diff])
        xlim = [x_min - x_diff/10, x_max + x_diff/10]
        ylim = [y_min - y_diff/10, y_max + y_diff/10]
        aspect = self.ellipse_VE * (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        self.set_axis_limits(bounds=xlim + ylim)
        
        self.window['axes'][0].set_aspect(aspect)
        self.window['axes'][0].invert_yaxis()
        # self.set_axis_limits(bounds=xlim + ylim)