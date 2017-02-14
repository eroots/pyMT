import numpy as np
from matplotlib.figure import Figure
import pyMT.utils as utils
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


class DataPlotManager(object):

    def __init__(self, fig=None):
        self.colour = ['darkgray', 'r', 'g', 'm', 'y', 'lime', 'peru', 'palegreen']
        self.sites = {'raw_data': [], 'data': [], 'response': []}
        self.toggles = {'raw_data': True, 'data': True, 'response': True}
        self.marker = {'raw_data': 'o', 'data': 'oo', 'response': '-'}
        self.errors = 'mapped'
        self.components = ['ZXYR']
        self.linestyle = '-'
        self.scale = 'sqrt(periods)'
        self.mec = 'k'
        self.markersize = 10
        self.edgewidth = 2
        self.sites = None
        self.tiling = [0, 0]
        self.show_outliers = True
        self.outlier_thresh = 2
        self.artist_ref = {'raw_data': [], 'data': [], 'response': []}
        self.y_labels = {'r': 'Apparent Resistivity', 'z': 'Impedance',
                         't': 'Transfer Function', 'p': 'Phase'}
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
        elif self.components[0][0].upper() == 'R':
            units = r'${\Omega}$-m'
        elif self.components[0][0].upper() == 'P':
            units = 'Degrees'
        if self.scale.lower() == 'periods' and not any(sub in self.components[0].lower() for sub in ('rho', 'pha')):
            units = ''.join(['s*', units])
        elif self.scale.lower() == 'sqrt(periods)' and not any(sub in self.components[0].lower() for sub in ('rho', 'pha')):
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
            self.axes[axnum].set_ylabel('{} ({})'.format(self.y_labels[self.components[0][0].lower()], self.units))
        if axnum + 1 > cols * (rows - 1):
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
            print('Redrawing Axes')
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
                    print(site.name)
                    self.set_labels(axnum=ii, site_name=site.name)
        self.set_bounds(Max=Max, Min=Min, axnum=list(range(ii + 1)))
        self.set_legend()
        # plt.show()

    def set_bounds(self, Max, Min, axnum):
        try:
            for ax in axnum:
                axrange = Max[ax] - Min[ax]
                self.axes[ax].set_ylim([Min[ax] - axrange / 4, Max[ax] + axrange / 4])
        except TypeError:
            axrange = Max - Min
            self.axes[axnum].set_ylim([Min - axrange / 4, Max + axrange / 4])

    def plot_site(self, site, Type='Data', ax=None):
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
            Err = toplotErr = None
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
                    toplot, toplotErr = utils.compute_phase(site, calc_comp=comp, errtype=errtype)
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
                artist = ax.errorbar(np.log10(site.periods), toplot, xerr=None,
                                     yerr=toplotErr, marker=marker,
                                     linestyle=linestyle, color=self.colour[ii],
                                     mec=self.mec, markersize=self.markersize,
                                     mew=edgewidth, picker=3)
                if self.show_outliers:
                    ma.append(max(toplot))
                    mi.append(min(toplot))
                else:
                    showdata = self.remove_outliers(toplot)
                    ma.append(max(showdata))
                    mi.append(min(showdata))
            except KeyError:
                # raise(e)
                artist = ax.text(0, 0, 'No Data')
                ma.append(0)
                mi.append(0)
            if Type == 'data':
                ax.aname = 'data'
            elif Type == 'raw_data':
                ax.aname = 'raw_data'
        # ax.set_title(site.name)
        return ax, max(ma), min(mi), artist

    def remove_outliers(self, data):
        nper = len(data)
        inds = []
        for idx, datum in enumerate(data):
            expected = 0
            for jj in range(max(1, idx - 2), min(nper, idx + 2)):
                expected += data[jj]
            expected /= jj
            tol = abs(self.outlier_thresh * expected)
            diff = datum - expected
            if abs(diff) > tol:
                inds.append(idx)
        return np.array([x for (idx, x) in enumerate(data) if idx not in inds])
