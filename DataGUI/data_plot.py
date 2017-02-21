# Main module for inversion data / response plotting. GUI is launched using 'python data_plot.py'.
# Currently a flagless launch looks for a file 'pystart' which contains the paths of the files to use.
# Format of the 'pystart' file is as follows:
# % Dataset_name  <--- '%' sign indicates that this line contains the dataset name.
# path /path/to/files   <--- a line containing the common path to the files that follow. Not needed,
#                            but if it is not specified then the path must be given for each file.
# list list_file_name   <--- Name of the file containing the site names for the given dataset.
#                            Same file as used for input to j2ws3d
# data data_file_name   <--- Name of the file containing the data as used for input to wsinv3dmt
# resp resp_file_name   <--- Name of the file containing the model responses, as output from wsinv3dmt
# raw  /path/to/dat/files <--- if the .dat files used are not in the same folder as the list file,
#                              this option tells the the program where to find them.
#
# Multiple datasets may be specified in a single startup file.
# Any combination of data / raw data / response may be used, as long as at least one is present.
# This module is callable with several flags, including a help flag, -h
# call 'python data_plot.py -h' for information on the other flags.
# EXAMPLE USAGE (command line entry)
# python data_plot.py -n abi-gren -c abi0:abi45:abi90
# |_________________  ||_________| |_________________|
#    Calls module   Specifies startup With startup file, selects datasets
#                    file 'abi-gren'     abi0, abi45 and abi90. If this 
#                                    flag is not given, all datasets are selected
import numpy as np
import re
from PyQt4.uic import loadUiType
from PyQt4 import QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import sys
import os
from pyMT import gplot, utils, data_structures

path = os.path.dirname(os.path.realpath(__file__))
Ui_MainWindow, QMainWindow = loadUiType(os.path.join(path, 'data_plot.ui'))
UiPopupMain, QPopupWindow = loadUiType(os.path.join(path, 'saveFile.ui'))


# ========================= #


class DataMain(QMainWindow, Ui_MainWindow):
    """
    Main GUI window for data plotting and manipulation.
    """

    def __init__(self, dataset_dict):
        super(DataMain, self).__init__()
        self.pick_tol = 0.15
        self.DEBUG = True
        self.setupUi(self)
        self.cid = {'DataSelect': []}
        # Holds the data for any sites that are removed during GUI execution so they can be added later.
        # Has the format of {site_name: {data_type: site}}
        self.stored_sites = {}
        self.old_val = ''
        self.fig = Figure()
        self.map = {'fig': None, 'canvas': None, 'axis': None,
                    'plots': {'all': None, 'highlight': None, 'mesh': None}}
        for ii, (dname, files) in enumerate(dataset_dict.items()):
            files = {file_type: files.get(file_type, '')
                     for file_type in ('data', 'raw_path', 'response', 'list')}
            dataset = data_structures.Dataset(listfile=files['list'],
                                              datafile=files['data'],
                                              responsefile=files['response'],
                                              datpath=files['raw_path'])
            if ii == 0:
                self.stored_datasets = {dname: dataset}
                self.dataset = dataset
                self.current_dataset = dname
            else:
                self.stored_datasets.update({dname: dataset})
        self.site_names = self.dataset.data.site_names[:6]
        self.setup_widgets()
        self.init_dpm()
        self.update_error_tree()
        self.addmpl(self.dpm.fig)
        self.dpm.plot_data()
        self.expand_tree_nodes(to_expand=self.site_names, expand=True)
        # Connect the error tree after so it doesn't run during it's init
        self.error_tree.itemChanged.connect(self.post_edit_error)
        self.update_comp_list()
        self.stored_key_presses = []

    @property
    def error_type(self):
        if self.dataErrRadio.isChecked():
            return 'raw'
        elif self.usedErrRadio.isChecked():
            return 'mapped'
        elif self.noErrRadio.isChecked():
            return 'none'

    @property
    def site_names(self):
        return self._site_names

    @site_names.setter
    def site_names(self, names):
        self._site_names = names
        if self.map['fig']:
            self.draw_map()

    def update_comp_list(self):
        ordered_comps = [comp for comp in self.dataset.data.ACCEPTED_COMPONENTS
                         if comp in self.dataset.data.components]
        c = 0
        if 'ZXXR' in ordered_comps:
            ordered_comps.append('RhoXX')
            ordered_comps.append('PhaXX')
            c += 1
        if 'ZXYR' in ordered_comps:
            ordered_comps.append('RhoXY')
            ordered_comps.append('PhaXY')
            c += 1
        if 'ZYYR' in ordered_comps:
            ordered_comps.append('RhoYY')
            ordered_comps.append('PhaYY')
            c += 1
        if 'ZYXR' in ordered_comps:
            ordered_comps.append('RhoYX')
            ordered_comps.append('PhaYX')
            c += 1
        if c == 4:
            ordered_comps.append('RhoDet')
            ordered_comps.append('PhaDet')

        self.comp_list.addItems(ordered_comps)
        labels = [self.comp_list.item(x) for x in range(self.comp_list.count())]
        labels = next(label for label in labels if label.text() in self.dpm.components)
        self.comp_list.setItemSelected(labels, True)

    def list_click(self):
        # This doesn't delete comps that are removed from the list
        # I think I need a way to keep track of what is contained in each
        # axis / line so that I can just modify / delete lines as need be,
        # rather than redrawing every time. The only time an axes needs to be
        # redrawn from scratch is if the site changes (although min/max have to
        # be updated if a new comp is added.)
        comps = [x.text() for x in self.comp_list.selectedItems()]
        if not all([x[0] == comps[0][0] for x in comps]):
            QtGui.QMessageBox.question(self, 'Message',
                                             'Can\'t mix components with different units',
                                             QtGui.QMessageBox.Ok)
            comps = self.dpm.components
            for ii in range(self.comp_list.count()):
                item = self.comp_list.item(ii)
                if item.text() in comps:
                    self.comp_list.setItemSelected(item, True)
                else:
                    self.comp_list.setItemSelected(item, False)
            return
        else:
            self.dpm.components = comps
            self.update_dpm(updated_sites=self.dpm.site_names,
                            updated_comps=self.dpm.components)

    @property
    def dTypes(self):
        return list(filter(None,
                           ['raw_data' * bool(self.toggleRaw.checkState()),
                            'data' * bool(self.toggleData.checkState()),
                            'response' * bool(self.toggleResponse.checkState())]))

    def init_dpm(self):
        self.dpm = gplot.DataPlotManager(fig=self.fig)
        self.dpm.sites = self.dataset.get_sites(site_names=self.site_names,
                                                dTypes=self.dTypes)
        self.dpm.scale = self.scalingBox.currentText()

    def setup_widgets(self):
        self.select_points_button.clicked.connect(self.select_points)
        self.select_points = False
        # self.error_table.itemChanged.connect(self.change_errmap)
        self.error_tree.itemDoubleClicked.connect(self.edit_error_tree)
        self.BackButton.clicked.connect(self.Back)
        self.ForwardButton.clicked.connect(self.Forward)
        self.WriteDataButton.clicked.connect(self.WriteData)
        self.comp_list.itemSelectionChanged.connect(self.list_click)
        self.toggleRaw.clicked.connect(self.toggle_raw)
        self.toggleData.clicked.connect(self.toggle_data)
        self.toggleResponse.clicked.connect(self.toggle_response)
        self.scalingBox.currentIndexChanged.connect(self.change_scaling)
        self.DEBUG_BUTTON.clicked.connect(self.DEBUG_METHOD)
        self.printPeriods.clicked.connect(self.print_periods)
        self.toggleRaw.setCheckState(bool(self.dataset.raw_data.sites) * 2)
        self.toggleData.setCheckState(bool(self.dataset.data.sites) * 2)
        self.toggleResponse.setCheckState(bool(self.dataset.response.sites) * 2)
        self.azimuthEdit.editingFinished.connect(self.set_azimuth)
        self.azimuthEdit.setText(str(self.dataset.azimuth))
        self.removeSites.clicked.connect(self.remove_sites)
        self.addSites.clicked.connect(self.add_sites)
        if len(self.stored_datasets) > 1:
            self.removeSites.setEnabled(False)
            self.addSites.setEnabled(False)
        self.currentDataset.addItems([dsets for dsets in self.stored_datasets.keys()])
        self.currentDataset.currentIndexChanged.connect(self.change_dataset)
        self.siteList.addItems(self.dataset.data.site_names)
        self.numSubplots.setText(str(len(self.site_names)))
        self.numSubplots.editingFinished.connect(self.num_subplots)
        self.dataErrRadio.toggled.connect(self.dummy_update_dpm)
        self.usedErrRadio.toggled.connect(self.dummy_update_dpm)
        self.noErrRadio.toggled.connect(self.dummy_update_dpm)
        self.refreshErrorTree.clicked.connect(self.update_error_tree)
        self.showMap.clicked.connect(self.show_map)
        self.sortSites.addItems(['Default', 'West-East',
                                 'South-North', 'Clustering'])
        self.sortSites.currentIndexChanged.connect(self.sort_sites)
        self.showOutliers.clicked.connect(self.toggle_outliers)
        self.actionList_File.triggered.connect(self.WriteList)
        self.actionData_File.triggered.connect(self.WriteData)

    def sort_sites(self, index=None):
        if index is None:
            index = self.sortSites.currentIndex()
        self.dataset.sort_sites(self.sortSites.itemText(index))
        self.siteList.clear()
        self.siteList.addItems(self.dataset.data.site_names)

    def show_map(self):
        if not self.map['fig']:
            self.draw_map()
        self.map['canvas'].show()

    def draw_map(self):
        if not self.map['fig']:
            self.map['fig'] = Figure()
        if not self.map['canvas']:
            self.map['canvas'] = FigureCanvas(self.map['fig'])
        if not self.map['axis']:
            self.map['axis'] = self.map['fig'].add_subplot(111)
        else:
            self.map['axis'].clear()
        sites = []
        for index in range(self.siteList.count()):
            sites.append(self.siteList.item(index).text())
        # locs, _ = utils.center_locs(self.dataset.data.get_locs(site_list=sites))
        # clocs, _ = utils.center_locs(self.dataset.data.get_locs(site_list=self.site_names))
        locs = self.dataset.data.get_locs(site_list=sites, azi=self.dataset.data.azimuth)
        clocs = self.dataset.data.get_locs(site_list=self.site_names, azi=self.dataset.data.azimuth)
        # print(clocs)
        # ymesh, Ny = utils.generate_mesh(site_locs=locs[:, 1], min_x=self.yMeshMin.value(), DEBUG=False)
        # xmesh, Nx = utils.generate_mesh(site_locs=locs[:, 0], min_x=self.xMeshMin.value(), DEBUG=False)
        # y, x = np.meshgrid(ymesh, xmesh)
        # c = np.ones_like(x)
        # print(xmesh.shape, ymesh.shape)
        # print(x.shape, y.shape, c.shape)
        # self.map['axis'].pcolor(y, x, c, facecolor='none', edgecolor='k')
        self.map['plots']['all'] = self.map['axis'].plot(locs[:, 1], locs[:, 0], 'k+')
        self.map['plots']['highlight'] = self.map['axis'].plot(clocs[:, 1], clocs[:, 0], 'ro')
        for ii, (ix, iy) in enumerate(clocs):
            print(self.site_names[ii])
            print(ix, iy)
            self.map['axis'].annotate(self.site_names[ii], xy=(iy, ix))
        self.map['canvas'].draw()

    def dummy_update_dpm(self, toggled):
        if toggled:
            self.update_dpm(updated_sites=None, updated_comps=None, remove_sites=None)

    def set_data_toggles(self):
        if not self.dataset.has_dType('raw_data'):
            self.toggleRaw.setCheckState(0)
            self.dpm.toggles['raw_data'] = False
        if not self.dataset.has_dType('data'):
            self.toggleData.setCheckState(0)
            self.dpm.toggles['data'] = False
        if not self.dataset.has_dType('response'):
            self.toggleResponse.setCheckState(0)
            self.dpm.toggles['response'] = False

    def change_dataset(self, index):
        print(self.dataset.data.site_names[:6])
        dset = self.currentDataset.itemText(index)
        self.current_dataset = dset
        self.dataset = self.stored_datasets[dset]
        self.set_data_toggles()
        # Temporarily disconnect the error tree so it doesn't freak out when the data is updated
        # self.error_tree.itemChanged.disconnect()
        self.update_error_tree()
        self.sort_sites()
        self.back_or_forward_button(shift=0)
        self.expand_tree_nodes(to_expand=self.site_names, expand=True)

        # self.error_tree.itemChanged.connect(self.post_edit_error)

    def num_subplots(self):
        text = self.numSubplots.text()
        numplots = utils.validate_input(text, int)
        if numplots is False:
            self.numSubplots.setText(str(self.dpm.num_sites))
            return
        current_num = self.dpm.num_sites
        if numplots != current_num:
            num_to_add = numplots - current_num
            if num_to_add > 0:
                for dType in self.dpm.sites.keys():
                    if self.dataset.has_dType(dType):
                        # tmp_sites = self.site_names
                        sites_to_add = [site for site in self.dataset.data.site_names
                                        if site not in self.site_names][:num_to_add]
                        for site in sites_to_add:
                            self.dpm.sites[dType].append(getattr(self.dataset, dType).sites[site])
                        # for ii in range(num_to_add):
                        #     site_name = next(site for site in self.dataset.data.site_names
                        #                      if site not in tmp_sites)
                        #     tmp_sites.append(site_name)
                        #     self.dpm.sites[dType].append(getattr(self.dataset, dType).sites[site_name])
            else:
                for dType in self.dpm.sites.keys():
                    if self.dpm.sites[dType]:
                        for ii in range(abs(num_to_add)):
                            self.dpm.sites[dType].pop(-1)
            self.site_names = self.dpm.site_names
            self.dpm.plot_data()
            self.dpm.fig.canvas.draw()

    def remove_sites(self):
        # Note that this method doesn't care about how many sites there are
        # relative to the number that should be plotted. I.E. in the edge case
        # that 6 sites are plotted but you remove some so that only 4 remain, there
        # will be an error.
        site_names = []
        for site in self.siteList.selectedItems():
            name = site.text()
            site_names.append(name)
            self.rmSitesList.addItem(self.siteList.takeItem(self.siteList.row(site)))
            self.stored_sites.update({name: {dType: getattr(self.dataset, dType).sites[name]
                                             for dType in self.dataset.data_types
                                             if self.dataset.has_dType(dType)}})
        self.dataset.remove_sites(sites=site_names)
        intersect = set(site_names).intersection(set(self.dpm.site_names))
        idx_plotted = list(sorted([self.dataset.data.site_names.index(x) for x in self.dpm.site_names
                                   if x in self.dataset.data.site_names]))
        if intersect:
            max_idx = len(self.dataset.data.site_names) - 1
            if idx_plotted[-1] == max_idx:
                direction = -1
                start = idx_plotted[-1]
                end = 0
            else:
                direction = 1
                start = idx_plotted[0]
                end = max_idx
            to_add = []
            new_sites = []
            for ii, site in enumerate(intersect):
                dont_add_these = set(idx_plotted) | set(to_add)
                to_add.append(next(x for x in range(start, end, direction)
                                   if x not in dont_add_these))
                new_sites.append(self.dataset.data.site_names[to_add[ii]])
            nsites = self.dataset.get_sites(site_names=new_sites, dTypes='all')
            self.dpm.replace_sites(sites_out=intersect, sites_in=nsites)
            self.site_names = self.dpm.site_names
            self.dpm.fig.canvas.draw()
            self.expand_tree_nodes(to_expand=self.site_names, expand=True)
        else:
            # Sites were removed, the map should be updated
            self.draw_map()

    def add_sites(self):
        # This method and the relevent methods in ws.data_structures are
        # not optimized. Adding many sites at once could be slow.
        for site in self.rmSitesList.selectedIndexes():
            name = self.rmSitesList.item(site.row()).text()
            self.siteList.addItem(self.rmSitesList.takeItem(site.row()))
            self.dataset.add_site(self.stored_sites[name])
            del self.stored_sites[name]
        self.draw_map()

    def print_periods(self):
        periods = list(self.dataset.raw_data.narrow_periods.keys())
        periods.sort()
        pretty_periods = [(per, self.dataset.raw_data.narrow_periods[per]) for
                          per in periods]
        print('{:>15} {:>15} {:>15}'.format('Period', 'Log(Period)', 'Perc'))
        for t in pretty_periods:
            k, v = t
            log_k = np.log10(k)
            if k in utils.truncate(self.dataset.data.periods):
                yn = '*'
            else:
                yn = ''
            if k < 1:
                k = -1 / k
            print('{:15.5} {:15.5} {:15.5}   {}'.format(k, log_k, v, yn))
        for t in self.dataset.data.periods:
            if t < 1:
                t = -1 / t
            print('{}\n'.format(utils.truncate(t)))

    def DEBUG_METHOD(self):
        # print(self.dpm.scale)
        # print(self.current_dataset)
        # print(self.stored_datasets)
        # print(self.dataset)
        # print(self.dataset.azimuth, self.dataset.data.azimuth,
        #       self.dataset.response.azimuth, self.dataset.raw_data.azimuth)
        print((self.dataset.data.sites[
              self.dataset.data.site_names[0]].flags))

    def set_azimuth(self):
        text = self.azimuthEdit.text()
        azi = utils.validate_input(text, float)
        if azi is False:
            self.azimuthEdit.setText(str(self.dataset.azimuth))
            return
        if azi != self.dataset.azimuth:
            if azi <= 0:
                azi += 360
                self.azimuthEdit.setText(str(azi))
            self.dataset.rotate_sites(azi=azi)
            self.update_dpm()
            self.draw_map()

    def change_scaling(self, index):
        self.dpm.scale = self.scalingBox.itemText(index).lower()
        self.update_dpm()
        for axnum, site_name in enumerate(self.site_names):
            cols = self.dpm.tiling[1]
            if axnum % cols == 0:
                self.dpm.set_labels(axnum, site_name)

    def toggle_outliers(self, event):
        self.dpm.show_outliers = event
        self.update_dpm()

    def toggle_raw(self, event):
        if self.dataset.raw_data.sites:
            if any([self.toggleRaw.checkState(),
                    self.toggleData.checkState(),
                    self.toggleResponse.checkState()]):
                self.dpm.toggles['raw_data'] = event
                self.update_dpm()
            # This can be used for quicker re-drawing since it just makes the data invisible,
            # however it doesn't properly update when the sites being plotted changes.
            # for axnum in range(len(self.dpm.axes)):
            #     artist = self.dpm.artist_ref['raw_data'][axnum][0]
            #     artist.set_visible(event)
            # self.dpm.fig.canvas.draw()
            else:
                self.toggleRaw.setCheckState(2)
        else:
            self.toggleRaw.setCheckState(0)

    def toggle_data(self, event):
        if self.dataset.data.sites:
            if any([self.toggleRaw.checkState(),
                    self.toggleData.checkState(),
                    self.toggleResponse.checkState()]):
                self.dpm.toggles['data'] = event
                self.update_dpm()
            else:
                self.toggleData.setCheckState(2)
        else:
            self.toggleData.setCheckState(0)

    def toggle_response(self, event):
        if self.dataset.response.sites:
            if any([self.toggleRaw.checkState(),
                    self.toggleData.checkState(),
                    self.toggleResponse.checkState()]):
                self.dpm.toggles['response'] = event
                self.update_dpm()
            else:
                self.toggleResponse.setCheckState(2)
        else:
            self.toggleResponse.setCheckState(0)

    def WriteData(self):
        keep_going = True
        while keep_going:
            outfile, ret = MyPopupDialog.get_file()
            if not ret or not outfile:
                break
            if ret and outfile:
                retval = self.dataset.write_data(outfile=outfile)
            if retval:
                break
            else:
                reply = QtGui.QMessageBox.question(self, 'Message',
                                                   'File already Exists. Overwrite?',
                                                   QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
                if reply == QtGui.QMessageBox.Yes:
                    self.dataset.write_data(outfile=outfile, overwrite=True)
                    break

    def WriteList(self):
        keep_going = True
        while keep_going:
            outfile, ret = MyPopupDialog.get_file()
            if not ret or not outfile:
                break
            if ret and outfile:
                retval = self.dataset.write_list(outfile=outfile)
            if retval:
                break
            else:
                reply = QtGui.QMessageBox.question(self, 'Message',
                                                   'File already Exists. Overwrite?',
                                                   QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
                if reply == QtGui.QMessageBox.Yes:
                    self.dataset.write_list(outfile=outfile, overwrite=True)
                    break

    def Back(self):
        """
        Bulk shifts the plotted sites back by the number of subplots.
        Works best if plotting sites as they appear in the list.
        """
        self.back_or_forward_button(shift=-1)
        # self.draw_map()

    def Forward(self):
        self.back_or_forward_button(shift=1)
        # self.draw_map()

    def back_or_forward_button(self, shift):
        self.site_names = self.shift_site_names(shift=shift)
        # If the sites haven't changed
        if set(self.site_names) == set(self.dpm.site_names) and shift != 0:
            return
        sites = self.dataset.get_sites(site_names=self.site_names, dTypes='all')
        self.dpm.replace_sites(sites_in=sites, sites_out=self.dpm.site_names)
        self.expand_tree_nodes(to_expand=self.dpm.site_names, expand=False)
        self.dpm.fig.canvas.draw()
        self.expand_tree_nodes(to_expand=self.site_names, expand=True)

    def shift_site_names(self, shift=1):
        try:
            idx_plotted = [self.dataset.data.site_names.index(x) for x in self.dpm.site_names]
        except ValueError:
            idx_plotted = list(range(len(self.dpm.site_names)))
        idx_toplot = []
        num_plots = len(idx_plotted)
        for ii, idx in enumerate(sorted(idx_plotted)):
            if ii == 0:
                idx_toplot.append(max([0, idx + (shift * num_plots)]))
            else:
                idx_toplot.append(max([0, idx_toplot[-1] + 1]))
        num_sites = len(self.dataset.data.site_names)
        if any(x > num_sites - 1 for x in idx_toplot):
            overflow = idx_toplot[-1] - num_sites
            idx_toplot = [x - overflow - 1 for x in idx_toplot]
        return [self.dataset.data.site_names[idx] for idx in idx_toplot]

    @utils.enforce_input(updated_sites=list, updated_comps=list, remove_sites=list)
    def update_dpm(self, updated_sites=None,
                   updated_comps=None, remove_sites=None):
        """
        This site updates axes given by the sites in updated_sites. If you want to replace
        a site with another, you must first call the replace_site method of the DataPlotManager
        """
        # dpm_sites = [site.name for site in self.dpm.sites['data']]  # Sites currently plotted
        self.dpm.errors = self.error_type
        if updated_sites is None:
            updated_sites = self.dpm.site_names
        if updated_comps is None:
            updated_comps = self.dpm.components
        dpm_sites = self.dpm.site_names
        updated_sites = list(set(updated_sites).intersection(set(dpm_sites)))  # Sites to be updated

        # print(updated_sites)
        # print([site.name for site in self.dpm.sites['data']])
        for site_name in updated_sites:
            # ind = next(ii for ii, site in
            #            enumerate(self.dpm.sites['data']) if site.name == site_name)
            ind = next(ii for ii, site in
                       enumerate(self.dpm.site_names) if site == site_name)
            self.dpm.sites['data'][ind] = self.dataset.data.sites[site_name]
            # print('single axes time')
            # t = time.time()
            self.dpm.redraw_single_axis(site_name=site_name, axnum=ind)
            # print(time.time() - t)
        # print(time.time() - t)
        # t = time.time()
        self.dpm.fig.canvas.draw()
        # print(time.time() - t)

    def edit_error_tree(self, column):
        """
        Is triggered when a cell in the error tree is double clicked, and determines
        whether or not that cell is editable. If it is, it allows you to edit the item
        which then calls post_edit_error.
        """
        self.stored_key_presses = QtGui.QApplication.keyboardModifiers()
        self.check_key_presses(verbose=True)
        item = self.error_tree.itemFromIndex(self.error_tree.selectedIndexes()[0])
        column = self.error_tree.currentColumn()
        if column >= 2 and item.flags():
            self.current_tree_item = item
            self.current_tree_col = column
            self.old_val = item.text(column)
            self.error_tree.editItem(item, column)

    def check_key_presses(self, verbose=False):
        msg = ''
        retval = {'Control': False, 'Shift': False, 'Alt': False}
        if self.stored_key_presses == QtCore.Qt.ShiftModifier:
            retval['Shift'] = True
        elif self.stored_key_presses == QtCore.Qt.ControlModifier:
            retval['Control'] = True
        elif self.stored_key_presses == QtCore.Qt.AltModifier:
            retval['Alt'] = True
        elif self.stored_key_presses == (QtCore.Qt.ControlModifier |
                                         QtCore.Qt.ShiftModifier):
            retval['Shift'] = True
            retval['Control'] = True
        elif self.stored_key_presses == (QtCore.Qt.AltModifier |
                                         QtCore.Qt.ShiftModifier):
            retval['Alt'] = True
            retval['Shift'] = True
        elif self.stored_key_presses == (QtCore.Qt.ControlModifier |
                                         QtCore.Qt.AltModifier):
            retval['Control'] = True
            retval['Alt'] = True
        elif self.stored_key_presses == (QtCore.Qt.ControlModifier |
                                         QtCore.Qt.ShiftModifier |
                                         QtCore.Qt.AltModifier):
            retval['Shift'] = True
            retval['Alt'] = True
            retval['Control'] = True
        if any(retval.values()):
            msg = '+'.join([k for k in retval if retval[k]])
            msg = msg.replace('Shift', 'Periods')
            msg = msg.replace('Control', 'Components')
            msg = msg.replace('Alt', 'Sites')
        if verbose and msg:
            print(' '.join(['Editting all', msg]))
        return retval

    def update_error_tree(self):
        try:
            self.error_tree.itemChanged.disconnect()
        except TypeError:
            tree_connected = False
        else:
            tree_connected = True
        self.error_tree.clear()
        ordered_comps = [comp for comp in self.dataset.data.ACCEPTED_COMPONENTS
                         if comp in self.dataset.data.components]
        header = [['Site'], ['Period'], ordered_comps]
        header = [item for sublist in header for item in sublist]
        # QtGui.QTreeWidgetItem(['Site', 'Period', 'Component'])
        self.error_tree.setColumnCount(len(header))
        self.error_tree.setHeaderItem(QtGui.QTreeWidgetItem(header))
        periods = sorted(self.dataset.data.periods)
        # for site in self.site_names: # Gives the tree for only the plotted sites
        self.tree_dict = {site: [] for site in self.dataset.data.site_names}
        for site in reversed(self.dataset.data.site_names):
            sitenode = QtGui.QTreeWidgetItem([site])
            self.tree_dict.update({site: sitenode})  # Stuff the site nodes into here
            self.error_tree.insertTopLevelItem(0, sitenode)
            for ii, p in enumerate(periods):
                # self.error_tree.insertTopLevelItem(1, QtGui.QTreeWidgetItem([str(p)]))
                # sitenode.addChild(QtGui.QTreeWidgetItem([str(p)]))
                if p < 1:
                    p = round(-1 / p, 1)
                pnode = QtGui.QTreeWidgetItem(sitenode, [str(p)])
                pnode.setFlags(QtCore.Qt.ItemIsEditable |
                               QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                pnode.setText(0, '')
                pnode.setText(1, str(p))
                for jj, comp in enumerate(ordered_comps):
                    pnode.setText(2 + jj, str(int(self.dataset.data.sites[site].errmap[comp][ii])))

        if tree_connected:
            self.error_tree.itemChanged.connect(self.post_edit_error)

    def expand_tree_nodes(self, to_expand, expand=True):
        # idx = [self.dataset.data.site_names.index(site) for site in to_expand]
        for site in to_expand:
            sitenode = self.tree_dict[site]
            if (sitenode.isExpanded() and not expand):
                sitenode.setExpanded(False)
            elif not sitenode.isExpanded() and expand:
                sitenode.setExpanded(True)

    def post_edit_error(self, item, column):
        # print(self.error_tree.columnCount())  # Number of components (+2 for site name and period)
        # print(self.error_tree.topLevelItemCount())  # of sites
        # print(item.childCount())
        # This method shouldn't call itself... There's probably a better way.
        self.error_tree.itemChanged.disconnect()
        if column != self.current_tree_col or item != self.current_tree_item:
            return
        key_presses = self.check_key_presses()
        comps = []
        site_names = []
        periods = []
        child_index = []
        parents = []
        initial_parent = item.parent()
        try:
            new_val = int(item.text(column))
        except ValueError:
            print('Value must be an integer')
            item.setText(column, self.old_val)
            return

        # Go over headers (Components)
        if key_presses['Control']:
            columns = range(2, self.error_tree.columnCount())
        # Or just get the current columns
        else:
            columns = [column]
        # Go over all of the periods
        if key_presses['Shift']:
            child_index = range(initial_parent.childCount())
        else:
            child_index = [item.parent().indexOfChild(item)]
        # Go over all sites
        if key_presses['Alt']:
            parents = self.tree_dict.values()
        else:
            parents = [initial_parent]
        for parent in parents:
            for idx in child_index:
                item = parent.child(idx)
                for col in columns:
                    # if bool(self.multiplicative.checkState()):
                    # set_val = min(new_val * int(item.text(col)), 9999)
                    # else:
                    # set_val = new_val
                    item.setText(col, str(new_val))
                    site_names.append(item.parent().text(0))
                    comps.append(self.error_tree.headerItem().text(col))
                    p = float(item.text(1))
                    if p < 0:
                        periods.append(-1 / p)
                    else:
                        periods.append(p)
        comps = set(comps)
        periods = set(periods)
        for site in set(site_names):
            self.dataset.data.sites[site].change_errmap(periods=periods, mult=new_val,
                                                        comps=comps,
                                                        multiplicative=False)

        self.error_tree.itemChanged.connect(self.post_edit_error)
        self.update_dpm(updated_sites=site_names, updated_comp=comps)

    def redraw(self, ax=None):
        pass
        # self.fig.redraw()

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)  # Make a canvas
        self.mplvl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(canvas=self.canvas,
                                         parent=self.mplwindow, coordinates=True)
        # Connect check box to instance
        self.canvas.draw()
        self.mplvl.addWidget(self.toolbar)

    def select_points(self):
        self.select_points = not(self.select_points)
        if not self.select_points:
            self.canvas.mpl_disconnect(self.cid['DataSelect'])
        else:
            self.cid['DataSelect'] = self.canvas.mpl_connect('pick_event', self.onpick)

    def onclick(self, event):
        if event.button == 1:
            print('Left mouse')
        if event.button == 2:
            print('Middle Mouse')
        if event.button == 3:
            print('Right Mouse')
        ax_index = next(ii for ii, ax in enumerate(self.dpm.axes) if ax == event.inaxes)
        if self.dpm.axes[ax_index].lines[2].contains(event):
            print('Yep, thats a point')
        if self.select_points:
            print(event.xdata, event.ydata)

    def onpick(self, pick_event):
        """
        Adds (on left click) or removes (on right click) periods from the data part of the dataset.
        Needs to be fixed so that the error tree is updated when a period is added. This should only
        add the new period, rather than do a full re-write of the tree.
        """
        # Due to a bug with mpl, if you pick an axis that has a legend outside, it double picks.
        event = pick_event.mouseevent
        ax_index = next(ii for ii, ax in enumerate(self.dpm.axes) if ax == event.inaxes)
        site_name = self.dpm.site_names[ax_index]
        ind = pick_event.ind[0]
        # period = 10 ** xdata
        isdata = False
        israw = False
        raw_site = self.dataset.raw_data.sites[site_name]
        data_site = self.dataset.data.sites[site_name]
        mec = pick_event.artist.properties()['markeredgewidth']
        linestyle = pick_event.artist.properties()['linestyle']
        if linestyle == self.dpm.marker['response']:
            if self.DEBUG:
                print('No action for response click.')
            return
        if mec == 0:
            israw = True
        if mec > 0:
            isdata = True
        if event.button == 1:
            if isdata:
                if self.DEBUG:
                    print('No action defined for data point')
                return
            if israw:
                period = float(raw_site.periods[ind])
                npd = data_site.periods[np.argmin(abs(data_site.periods - period))]
                if utils.percdiff(npd, period) < self.pick_tol:
                    if self.DEBUG:
                        print('There is already a period pretty close to that...')
                    return
                else:
                    toadd = self.dataset.raw_data.get_data(periods=period,
                                                           components=self.dataset.data.components,
                                                           lTol=None, hTol=None)
                    self.dataset.data.add_periods(toadd)
                    if self.DEBUG:
                        print('Adding period {}, freq {}'.format(period, 1 / period))
                    self.update_dpm(updated_sites=self.dpm.site_names,
                                    updated_comp=self.dataset.data.components)
        if event.button == 2:
            if israw:
                print(ind)
                period = float(raw_site.periods[ind])
                freq = 1 / period
                perc = self.dataset.raw_data.master_periods[period] * 100
                print('Period {} (Frequency {}) is available at {}% of sites'.format(
                      utils.truncate(period), utils.truncate(freq), utils.truncate(perc)))

        if event.button == 3:
            if israw:
                if self.DEBUG:
                    print('No action defined for raw data point')
                return
            if len(data_site.periods) == 1:
                print('Data must have at least one period in it!')
                return
            if isdata:
                period = float(data_site.periods[ind])
                npd = data_site.periods[np.argmin(abs(data_site.periods - period))]
                self.dataset.data.remove_periods(periods=period)
                if self.DEBUG:
                    print('Removing period {}, freq {}'.format(period, 1 / period))
                self.update_dpm()
        #     print('Right Mouse')
        # if self.dpm.axes[ax_index].lines[2].contains(event):
            # print('Yep, thats a point')
        # if self.select_points:
            # print(event.xdata, event.ydata)

    def disconnect_mpl_events(self):
        for key in self.cid.keys():
            # try:
            self.canvas.mpl_disconnect(self.cid[key])

    def connect_mpl_events(self):
        # Can change this later if I want to add something for clicking an axis and not a point
        if self.select_points:
            # self.cid['DataSelect'] = self.canvas.mpl_connect('button_press_event', self.onclick)
            self.cid['DataSelect'] = self.canvas.mpl_connect('pick_event', self.onpick)


class MyTableModel(QtCore.QAbstractTableModel):
    def __init__(self, datain, parent=None, *args):
        QtCore.QAbstractTableModel.__init__(self, parent, *args)
        self.arraydata = datain
        self.headerdata = None
        self.model = QtGui.QStandardItemModel(self)

    def rowCount(self, parent):
        return len(self.arraydata)

    def columnCount(self, parent):
        return len(self.arraydata[0])

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == QtCore.Qt.DisplayRole:
            i = index.row()
            j = index.column()
            return (self.arraydata[i][j])
        else:
            return None

    def setData(self, index, value, role):
        if self.flags(index):
            self.arraydata[index.row()][index.column()] = value
        return True

    def flags(self, index):
        if index.column() < 2:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | \
                QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class FileInputParser(object):
    # If I add functionality to add datasets after you've already opened a GUI,
    # I may want to make it so I can create an instance of this with a list
    # of accepted file types, and then use that instance to check every time something
    # is added.

    def __init__(self, files=''):
        pass

    @staticmethod
    def get_files_dialog():
        keep_going = True
        start_dict = {}
        while keep_going:
            startup, ret = MyPopupDialog.get_file(default='pystart', label='Startup:')
            if not ret or not startup:
                print('OK Fine, don''t use me...')
                return '', ret
            if ret and startup:
                try:
                    start_dict = FileInputParser.read_pystart(startup)
                except FileNotFoundError as e:
                    print('File {} not found. Try again.'.format(startup))
                else:
                    keep_going = False
        return start_dict, ret

    @staticmethod
    def verify_files(files):
        if files:
            for dataset_name, startup in files.items():
                for Type, file in startup.items():
                    if Type != 'raw_path':
                        if not utils.check_file(file):
                            print('File {} not found.'.format(file))
                            return False
                        # try:
                        #     with open(file, 'r'):
                        #         pass
                        # except FileNotFoundError as e:
                        #     print('File {} not found.'.format(file))
                        #     return False
            else:
                return True
        else:
            return False

    @staticmethod
    def read_pystart(startup):
        dset = ''
        acceptable = ('data', 'list', 'response', 'raw_path', 'model', 'path')
        abbreviations = {'raw': 'raw_path', 'resp': 'response', 'mod': 'model', 'lst': 'list'}
        try:
            with open(startup, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line[0] == '#' or line[0].strip() == '':
                        continue
                    elif line[0] == '%':
                        dname = [x.strip() for x in line.split('%') if x]
                        if dset == '':
                            dset = dname[0]
                            retval = {dset: {}}
                        else:
                            dset = dname[0]
                            retval.update({dset: {}})
                        continue
                    elif line[0] in '!@$^&*':
                        print('Unrecognized character {} in {}.'
                              ' Are you sure you don\'t mean %?'.format(line[0], startup))
                        return False
                    elif dset == '':
                        dset = 'dataset1'
                        retval = {dset: {}}
                    parts = line.split()
                    # Only check for abbreviations if the keyword isn't already present in the list
                    dType = parts[0].lower()
                    dType_abr = re.sub(r'\b' + '|'.join(abbreviations.keys()) + r'\b',
                                       lambda m: abbreviations[m.group(0)], parts[0])
                    if dType in acceptable:
                        retval[dset].update({dType: parts[1]})
                    elif dType_abr in acceptable:
                        retval[dset].update({dType_abr: parts[1]})
                    else:
                        print('Unrecognized keyword: {}'.format(dType))
            # If a dataset 'COMMON' is present, fill in any blanks in the other datasets
            # with its values
            # if 'common' in [dname.lower() for dname in retval.keys()]:
            # for dataset_files in
            for dataset_files in retval.values():
                if 'path' in dataset_files.keys():
                    path = dataset_files['path']
                    del dataset_files['path']
                    for file_type in dataset_files.keys():
                        if not os.path.dirname(dataset_files[file_type]):
                            dataset_files[file_type] = os.path.join(path, dataset_files[file_type])
        except FileNotFoundError:
            print('File not found: {}'.format(startup))
            retval = False
        return retval


# Should just modify this to allow for multiple lines of input, as well as a browse button.
# The returned files can then be parsed by FileInputParser.
class MyPopupDialog(UiPopupMain, QPopupWindow):
    """
    Creates a pop-up window belonging to parent

    Args:
        parent (obj): The parent window the created pop-up will be attached to
    """

    def __init__(self, parent=None):
        super(MyPopupDialog, self).__init__(parent)
        self.setupUi(self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    @staticmethod
    def get_file(parent=None, message='', default='', label='Output'):
        """Summary

        Args:
            parent (None, optional): Description
            message (str, optional): Description
            default (str, optional): Description
            label (str, optional): Description

        Returns:
            TYPE: Description
        """
        dialog = MyPopupDialog(parent)
        dialog.lineEdit.setText(default)
        dialog.label.setText(label)
        ret = dialog.exec_()
        file = dialog.lineEdit.text()
        return file, ret


def parse_commandline(args):
    start_file = None
    if '-d' in sys.argv:
        start_file = 'pystart'
    elif '-b' in sys.argv:
        start_file, response = FileInputParser.get_files_dialog()
    elif '-n' in sys.argv:
        idx = sys.argv.index('-n')
        try:
            start_file = sys.argv[idx + 1]
        except IndexError:
            print('You must name your pystart file when using the -n option!')
            return None
    elif '-h' in sys.argv:
        print('Options include: \n\t -d : Use default startup file "pystart"' +
              '\n\t -b : Browse for start file or data files (Not yet implemented)' +
              '\n\t -n : Specify the start file you wish to use' +
              '\n\t -l : List the dataset names present in the start file you have chosen' +
              '\n\t -c : Choose a specific dataset(s) listed within the start file you have chosen' +
              '\n\t\t For multiple datasets, separate names with a colon (:)')
        return None
    else:
        files, response = FileInputParser.get_files_dialog()
    # if start_file == '':
    #     start_file, response = FileInputParser.get_files_dialog()
    if isinstance(start_file, str):
        files = FileInputParser.read_pystart(start_file)
    if files is False:
        return None
    # If you made it this far, I assume you have a good starting file.
    if '-c' in sys.argv:
        idx = sys.argv.index('-c')
        if len(sys.argv) == idx + 1:
            print('You need to specify the model you want with the -c option.')
            return None
        dnames = sys.argv[idx + 1].split(':')
        dset = {dname: dataset for dname, dataset in files.items() if dname in dnames}
        if not dset:
            print('Dataset {} not found in {}'.format(sys.argv[idx + 1], start_file))
        else:
            files = dset
    elif '-l' in sys.argv:
        print('Datasets contained in {} are: {}'.format(start_file, list(files.keys())))
        print('Load all [y], Load one [Dataset name], or Exit [n]?')
        while True:
            user_input = input()
            if user_input.lower() == 'n':
                return None
            elif user_input.lower() == 'y':
                break
            elif user_input in files.keys():
                files = {dname: dataset for dname, dataset in files.items() if dname == user_input}
                break
            else:
                print('That was not one of the options...')
    return files


# If this is run directly, launch the GUI
def main():
    app = QtGui.QApplication(sys.argv)  # Starts GUI event loop
    # These 3 options are mutually exclusive, as they all decide how to get the files
    files = parse_commandline(sys.argv)
    if files is None:
        return
    # If no option is specified, or if you tried one and it failed
    verify = FileInputParser.verify_files(files)
    if verify:
        # Because files is a dictionary, the plotter may not load the same dataset first every time.
        # The dataset that gets loaded first should be the same given the same start file.
        mainGUI = DataMain(files)  # Instantiate a GUI window
        mainGUI.show()  # Show it.
        ret = app.exec_()
        sys.exit(ret)  # Properly close the loop when the window is closed.
        mainGUI.disconnect_mpl_events()
    print('Exiting')


if __name__ == '__main__':
    main()
    # print('Done.')
