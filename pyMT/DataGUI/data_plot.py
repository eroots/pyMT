#!/usr/bin/env python
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
from PyQt5.uic import loadUiType
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import sys
import os
from pyMT import gplot, utils, data_structures
from pyMT.GUI_common.classes import FileDialog

path = os.path.dirname(os.path.realpath(__file__))

Ui_MainWindow, QMainWindow = loadUiType(os.path.join(path, 'data_plot.ui'))
UiPopupMain, QPopupWindow = loadUiType(os.path.join(path, 'saveFile.ui'))
UI_MapViewWindow, QMapViewMain = loadUiType(os.path.join(path, 'map_viewer.ui'))


# ========================= #
class MapMain(QMapViewMain, UI_MapViewWindow):

    def __init__(self, dataset, sites, active_sites):
        super(MapMain, self).__init__()
        self.setupUi(self)
        self.fig = Figure()
        self.active_period = 0
        self.map = gplot.MapView(figure=self.fig)
        self.init_map(dataset, sites, active_sites)
        self.periods = self.map.site_data['data'].periods
        self.set_period_label()
        self.add_mpl(self.map.window['figure'])
        self.connect_widgets()

    def connect_widgets(self):
        self.DEBUG.clicked.connect(self.DEBUG_FUNC)
        #  Connect induction arrow toggles
        if self.map.dataset.data.inv_type in (3, 5):
            self.toggle_dataInduction.clicked.connect(self.update_map)
            self.toggle_responseInduction.clicked.connect(self.update_map)
            self.toggle_normalizeInduction.clicked.connect(self.update_map)
        else:
            self.toggle_dataInduction.setEnabled(False)
            self.toggle_responseInduction.setEnabled(False)
            self.toggle_normalizeInduction.setEnabled(False)
        #  Connect phase tensor toggles
        if self.map.dataset.data.inv_type in (1, 5):
            self.toggle_dataPhaseTensor.clicked.connect(self.data_phase_tensor)
            self.toggle_responsePhaseTensor.clicked.connect(self.resp_phase_tensor)
            self.toggle_nonePhaseTensor.clicked.connect(self.none_phase_tensor)
            self.PhaseTensor_fill.currentIndexChanged.connect(self.update_map)
        else:
            self.toggle_dataPhaseTensor.setEnabled(False)
            self.toggle_responsePhaseTensor.setEnabled(False)
            self.toggle_nonePhaseTensor.setEnabled(False)
            self.PhaseTensor_fill.setEnabled(False)
        #  Connect pseudo-section plotting toggles
        self.toggle_rhoPseudo.clicked.connect(self.update_map)
        self.toggle_phasePseudo.clicked.connect(self.update_map)
        self.toggle_dataPseudo.clicked.connect(self.update_map)
        self.toggle_responsePseudo.clicked.connect(self.update_map)
        self.Pseudosection_fill.currentIndexChanged.connect(self.update_map)
        self.nInterp.valueChanged.connect(self.update_map)
        #  Set up period scroll bar
        self.PeriodScrollBar.valueChanged.connect(self.change_period)
        self.PeriodScrollBar.setMinimum(1)
        self.PeriodScrollBar.setMaximum(len(self.map.site_data['data'].periods))
        #  Set up colour map selections
        self.groupColourmaps = QtWidgets.QActionGroup(self)
        self.groupColourmaps.triggered.connect(self.set_colourmap)
        self.actionJet.setActionGroup(self.groupColourmaps)
        self.actionJet_r.setActionGroup(self.groupColourmaps)
        self.actionJet_plus.setActionGroup(self.groupColourmaps)
        self.actionJet_plus_r.setActionGroup(self.groupColourmaps)
        self.actionBgy.setActionGroup(self.groupColourmaps)
        self.actionBgy_r.setActionGroup(self.groupColourmaps)
        self.actionBwr.setActionGroup(self.groupColourmaps)
        self.actionBwr_r.setActionGroup(self.groupColourmaps)
        # Set up colour limits
        self.actionRho_cax.triggered.connect(self.set_rho_cax)
        self.actionPhase_cax.triggered.connect(self.set_phase_cax)
        self.actionDifference_cax.triggered.connect(self.set_difference_cax)
        # Set up point / marker options
        self.groupAnnotate = QtWidgets.QActionGroup(self)
        self.actionAnnotate_All.setActionGroup(self.groupAnnotate)
        self.actionAnnotate_None.setActionGroup(self.groupAnnotate)
        self.actionAnnotate_Active.setActionGroup(self.groupAnnotate)
        self.groupAnnotate.triggered.connect(self.set_annotations)
        self.actionMarker_Size.triggered.connect(self.set_marker_size)
        self.actionMarker_Shape.triggered.connect(self.set_marker_shape)
        self.actionMarker_Colour.triggered.connect(self.set_marker_colour)
        self.actionFilled.triggered.connect(self.set_marker_fill)
        # RMS plotting
        if self.map.dataset.response.sites:
            self.plotRMS.clicked.connect(self.update_map)

    def data_phase_tensor(self):
        if self.toggle_dataPhaseTensor.isChecked():
            self.toggle_nonePhaseTensor.setCheckState(0)
        self.update_map()

    def resp_phase_tensor(self):
        if self.toggle_responsePhaseTensor.isChecked():
            self.toggle_nonePhaseTensor.setCheckState(0)
        self.update_map()

    def none_phase_tensor(self):
        if self.toggle_nonePhaseTensor.isChecked():
            self.toggle_dataPhaseTensor.setCheckState(0)
            self.toggle_responsePhaseTensor.setCheckState(0)
        self.update_map()

    def set_marker_fill(self):
        if self.actionFilled.isChecked():
            self.map.site_fill = True
        else:
            self.map.site_fill = False
        self.update_map()

    def set_annotations(self):
        if self.actionAnnotate_All.isChecked():
            self.map.annotate_sites = 'all'
        elif self.actionAnnotate_None.isChecked():
            self.map.annotate_sites = 'none'
        elif self.actionAnnotate_Active.isChecked():
            self.map.annotate_sites = 'active'
        self.update_map()

    def set_marker_size(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Marker Size',
                                                         'Value:',
                                                         self.map.markersize,
                                                         0.1, 20, 1)
        if ok_pressed and d != self.map.markersize:
            self.map.markersize = d
            self.update_map()

    def set_marker_colour(self):
        colours = ['Black', 'Blue', 'Red', 'Green', 'Cyan', 'Yellow', 'Magenta']
        codes = {'Black': 'k',
                 'Blue': 'b',
                 'Red': 'r',
                 'Green': 'g',
                 'Cyan': 'c',
                 'Yellow': 'y',
                 'Magenta': 'm'}
        d, ok_pressed = QtWidgets.QInputDialog.getItem(self,
                                                       'Marker Colour',
                                                       'Colour:',
                                                       colours,
                                                       0, False)
        if ok_pressed:
            code = codes[d]
            self.map.site_colour = code
            self.update_map()

    def set_marker_shape(self):
        shapes = ['.', 'o', '*', 'x', '^', 'v']
        d, ok_pressed = QtWidgets.QInputDialog.getItem(self,
                                                       'Marker Shape',
                                                       'Shape:',
                                                       shapes,
                                                       0, False)
        if ok_pressed:
            self.map.site_marker = d
            self.update_map()

    def set_rho_cax(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Lower limit',
                                                         'Value:',
                                                         self.map.rho_cax[0],
                                                         -100, 100, 2)
        if ok_pressed:
            lower = d
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Upper limit',
                                                         'Value:',
                                                         self.map.rho_cax[1],
                                                         -100, 100, 2)
        if ok_pressed:
            upper = d
        if lower < upper and [lower, upper] != self.map.rho_cax:
            self.map.rho_cax = [lower, upper]
            self.update_map()
        else:
            print('Invalid colour limits')

    def set_difference_cax(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Lower limit',
                                                         'Value:',
                                                         self.map.diff_cax[0],
                                                         -1000, 1000, 2)
        if ok_pressed:
            lower = d
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Upper limit',
                                                         'Value:',
                                                         self.map.diff_cax[1],
                                                         -1000, 1000, 2)
        if ok_pressed:
            upper = d
        if lower < upper and [lower, upper] != self.map.diff_cax:
            self.map.diff_cax = [lower, upper]
            self.update_map()
        else:
            print('Invalid colour limits')

    def set_phase_cax(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Lower limit',
                                                         'Value:',
                                                         self.map.phase_cax[0],
                                                         -180, 180, 2)
        if ok_pressed:
            lower = d
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Upper limit',
                                                         'Value:',
                                                         self.map.phase_cax[1],
                                                         -180, 180, 2)
        if ok_pressed:
            upper = d
        if lower < upper and [lower, upper] != self.map.phase_cax:
            self.map.phase_cax = [lower, upper]
            self.update_map()
        else:
            print('Invalid colour limits')

    def set_colourmap(self):
        if self.actionJet.isChecked():
            self.map.colourmap = 'jet'
        if self.actionJet_r.isChecked():
            self.map.colourmap = 'jet_r'
        if self.actionJet_plus.isChecked():
            self.map.colourmap = 'jet_plus'
        if self.actionJet_plus_r.isChecked():
            self.map.colourmap = 'jet_plus_r'
        if self.actionBgy.isChecked():
            self.map.colourmap = 'bgy'
        if self.actionBgy_r.isChecked():
            self.map.colourmap = 'bgy_r'
        if self.actionBwr.isChecked():
            self.map.colourmap = 'bwr'
        if self.actionBwr_r.isChecked():
            self.map.colourmap = 'bwr_r'
        self.update_map()

    def set_period_label(self):
        self.PeriodLabel.setText(' '.join([str(self.periods[self.active_period]), 's']))

    def change_period(self, idx):
        self.active_period = idx - 1
        self.set_period_label()
        self.update_map()

    def DEBUG_FUNC(self):
        print('This is a debug function')
        # print(self.map.site_locations['all'])
        # print(self.map.active_sites)
        # print(self.map.site_locations['active'])
        # print(self.map.site_names)
        X, Y = [], []
        for site in self.map.site_names:
            if 'TZXR' in self.map.site_data['data'].sites[site].components:
                X.append(self.map.site_data['data'].sites[site].data['TZXR'][-1])
            else:
                X.append(0)
            if 'TZYR' in self.map.site_data['data'].sites[site].components:
                Y.append(self.map.site_data['data'].sites[site].data['TZYR'][-1])
            else:
                Y.append(0)
        # arrows = np.transpose(np.array((X, Y)))
        # print(arrows)
        # induction_toggles = self.get_induction_toggles()
        # print(induction_toggles['data'])
        # print(induction_toggles['normalize'])

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(canvas=self.canvas,
                                         parent=self.mplwindow, coordinates=True)
        self.canvas.draw()
        self.mplvl.addWidget(self.toolbar)

    def init_map(self, dataset, sites, active_sites):
        self.map.dataset = dataset
        self.map.data = dataset.data
        self.map.raw_data = dataset.raw_data
        self.map.response = dataset.response
        self.map.site_names = sites
        self.map._active_sites = active_sites
        self.map.site_locations['generic'] = self.map.get_locations(
            sites=self.map.generic_sites)
        self.map.site_locations['active'] = self.map.get_locations(
            sites=self.map.active_sites)
        self.map.site_locations['all'] = self.map.get_locations(self.map.site_names)

    def update_map(self):
        # Currently redraws the whole map every time
        # This should be changed to just destroy and redraw whatever features are needed
        # print(self.map.site_locations['generic'])
        # Also there should be a mechanism that makes sure this is only redrawn if something changes
        self.map.window['axes'][0].clear()
        if self.map.window['colorbar']:
            self.map.window['colorbar'].remove()
            self.map.window['colorbar'] = None
        # DEBUG
        # print('I am updating the map')
        pseudosection_toggles = self.get_pseudosection_toggles()
        if pseudosection_toggles['data'] and pseudosection_toggles['fill']:
            fill_param = ''.join([pseudosection_toggles['fill'],
                                  pseudosection_toggles['component']])
            self.map.plan_pseudosection(data_type=pseudosection_toggles['data'],
                                        fill_param=fill_param,
                                        period_idx=self.active_period,
                                        n_interp=self.nInterp.value())
        self.map.plot_rms = self.plotRMS.checkState()
        self.map.plot_locations()
        PT_toggles = self.get_PT_toggles()
        if 'None' not in PT_toggles['data']:
            self.map.plot_phase_tensor(data_type=PT_toggles['data'],
                                       fill_param=PT_toggles['fill'],
                                       period_idx=self.active_period)
        induction_toggles = self.get_induction_toggles()
        if induction_toggles['data']:
            self.map.plot_induction_arrows(data_type=induction_toggles['data'],
                                           normalize=induction_toggles['normalize'],
                                           period_idx=self.active_period)
        self.toolbar.update()
        self.toolbar.push_current()
        # DEBUG
        # print('Updating Map')
        self.canvas.draw()

    def get_pseudosection_toggles(self):
        toggles = {'data': [], 'fill': None, 'component': None}
        if self.toggle_dataPseudo.checkState() and self.map.dataset.data.sites:
            toggles['data'].append('data')
        if self.toggle_responsePseudo.checkState() and self.map.dataset.data.sites:
            toggles['data'].append('response')
        if self.toggle_rhoPseudo.isChecked():
            toggles['fill'] = 'Rho'
        elif self.toggle_phasePseudo.isChecked():
            toggles['fill'] = 'Pha'
        index = self.Pseudosection_fill.currentIndex()
        toggles['component'] = self.Pseudosection_fill.itemText(index)
        return toggles

    def get_PT_toggles(self):
        toggles = {'data': [], 'fill': 'Alpha'}
        if self.toggle_dataPhaseTensor.checkState()and self.map.dataset.data.sites:
            toggles['data'].append('data')
        if self.toggle_responsePhaseTensor.checkState()and self.map.dataset.response.sites:
            toggles['data'].append('response')
        if self.toggle_nonePhaseTensor.checkState() or not toggles['data']:
            toggles['data'].append('None')
            self.toggle_nonePhaseTensor.setCheckState(2)
            self.toggle_dataPhaseTensor.setCheckState(0)
            self.toggle_responsePhaseTensor.setCheckState(0)
        index = self.PhaseTensor_fill.currentIndex()
        toggles['fill'] = self.PhaseTensor_fill.itemText(index)
        return toggles

    def get_induction_toggles(self):
        toggles = {'data': [], 'normalize': False}
        if self.toggle_dataInduction.checkState() and self.map.dataset.data.sites:
            toggles['data'].append('data')
        else:
            self.toggle_dataInduction.setCheckState(0)
        if self.toggle_responseInduction.checkState() and self.map.dataset.response.sites:
            toggles['data'].append('response')
        else:
            self.toggle_responseInduction.setCheckState(0)
        if self.toggle_normalizeInduction.checkState():
            toggles['normalize'] = True

        return toggles


class DataMain(QMainWindow, Ui_MainWindow):
    """
    Main GUI window for data plotting and manipulation.
    """

    def __init__(self, dataset_dict):
        super(DataMain, self).__init__()
        self.fig_dpi = 300
        self.pick_tol = 0.15
        self.DEBUG = False
        self.setupUi(self)
        self.cid = {'DataSelect': []}
        # Holds the data for any sites that are removed during GUI execution so they can be added later.
        # Has the format of {site_name: {data_type: site}}
        self.stored_sites = {}
        self.old_val = ''
        self.fig = Figure()
        self.file_dialog = FileDialog(self)
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
        # self.update_comp_list()
        self.update_comp_table()
        if self.dataset.rms:
            self.init_rms_tables()
        self.stored_key_presses = []
        self.map_view = MapMain(dataset=self.dataset,
                                active_sites=self.site_names,
                                sites=self.dataset.data.site_names)

    def init_rms_tables(self):
        ordered_comps = [comp for comp in self.dataset.data.ACCEPTED_COMPONENTS
                         if comp in self.dataset.data.components]
        header = ['Total'] + ordered_comps
        periods = ['Total'] + list(self.dataset.data.periods)
        self.stationRMS.setRowCount(self.dataset.data.NS)
        self.stationRMS.setColumnCount(self.dataset.data.NR + 1)
        self.periodRMS.setRowCount(self.dataset.data.NP + 1)
        self.periodRMS.setColumnCount(self.dataset.data.NR + 1)
        for ii, label in enumerate(header):
            self.stationRMS.setHorizontalHeaderItem(ii, QtWidgets.QTableWidgetItem(label))
            self.periodRMS.setHorizontalHeaderItem(ii, QtWidgets.QTableWidgetItem(label))
            for jj, site in enumerate(self.dataset.data.site_names):
                if ii == 0:
                    self.stationRMS.setVerticalHeaderItem(jj, QtWidgets.QTableWidgetItem(site))
                node = QtWidgets.QTableWidgetItem(str(self.dataset.rms['Station'][site][label])[:4])
                node.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.stationRMS.setItem(jj, ii, node)
            for jj, period in enumerate(periods):
                if jj > 0:
                    if period < 1:
                        period = utils.truncate(- 1 / period)
                    period = str(period)
                if ii == 0:
                    self.periodRMS.setVerticalHeaderItem(jj, QtWidgets.QTableWidgetItem(str(period)))
                if jj == 0:
                    node = QtWidgets.QTableWidgetItem(str(self.dataset.rms['Component'][label])[:4])
                    node.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    self.periodRMS.setItem(jj, ii, QtWidgets.QTableWidgetItem(node))
                else:
                    node = QtWidgets.QTableWidgetItem(str(self.dataset.rms['Period'][label][jj - 1])[:4])
                    node.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    self.periodRMS.setItem(jj, ii, QtWidgets.QTableWidgetItem(node))
        for ii in range(len(header)):
            self.stationRMS.horizontalHeader().setSectionResizeMode(ii,
                                                                    QtWidgets.QHeaderView.ResizeToContents)
            self.periodRMS.horizontalHeader().setSectionResizeMode(ii,
                                                                   QtWidgets.QHeaderView.ResizeToContents)

    def update_rms_info(self):
        pass
        # ordered_comps = [comp for comp in self.dataset.data.ACCEPTED_COMPONENTS
        #                  if comp in self.dataset.data.components]

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
        # if self.map['fig']:
        #     self.draw_map()

    def update_comp_table(self):
        ordered_comps = [comp for comp in self.dataset.data.ACCEPTED_COMPONENTS
                         if comp in self.dataset.data.components]
        c = 0
        all_comps = {'Impedance': ordered_comps,
                     'Rho': [],
                     'Phase': [],
                     'Bostick': []}
        if 'TZXR' in ordered_comps:
            all_comps.update({'Tipper': [comp for comp in ordered_comps if comp[0].upper() == 'T']})
            ordered_comps.remove('TZXR')
            ordered_comps.remove('TZXI')
            ordered_comps.remove('TZYR')
            ordered_comps.remove('TZYI')
        if 'ZXXR' in ordered_comps:
            all_comps['Rho'].append('RhoXX')
            all_comps['Phase'].append('PhaXX')
            all_comps['Bostick'].append('BostXX')
            c += 1
        if 'ZXYR' in ordered_comps:
            all_comps['Rho'].append('RhoXY')
            all_comps['Phase'].append('PhaXY')
            all_comps['Bostick'].append('BostXY')
            c += 1
        if 'ZYYR' in ordered_comps:
            all_comps['Rho'].append('RhoYY')
            all_comps['Phase'].append('PhaYY')
            all_comps['Bostick'].append('BostYY')
            c += 1
        if 'ZYXR' in ordered_comps:
            all_comps['Rho'].append('RhoYX')
            all_comps['Phase'].append('PhaYX')
            all_comps['Bostick'].append('BostYX')
            c += 1
        if c == 4:
            all_comps['Rho'].append('RhoDet')
            all_comps['Rho'].append('RhoAAV')
            all_comps['Rho'].append('RhoGAV')
            all_comps['Phase'].append('PhaDet')
            all_comps['Phase'].append('PhaAAV')
            all_comps['Phase'].append('PhaGAV')
            all_comps['Bostick'].append('BostDet')
            all_comps['Bostick'].append('BostAAV')
            all_comps['Bostick'].append('BostGAV')
        header = ['Impedance', 'Rho', 'Phase', 'Bostick']
        if 'Tipper' in all_comps.keys():
            header.insert(1, 'Tipper')
        self.comp_table.setColumnCount(len(header))
        max_len = max([len(comp) for comp in all_comps.values()])
        self.comp_table.setRowCount(max_len)
        for ii, label in enumerate(header):
            self.comp_table.setHorizontalHeaderItem(ii, QtWidgets.QTableWidgetItem(label))
        for col, dtype in enumerate(header):
            for row, comp in enumerate(all_comps[dtype]):
                node = QtWidgets.QTableWidgetItem(all_comps[dtype][row])
                node.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.comp_table.setItem(row, col, node)
        for ii in range(len(header)):
            self.comp_table.horizontalHeader().setSectionResizeMode(ii,
                                                                    QtWidgets.QHeaderView.ResizeToContents)
        for ii in range(max_len):
            self.comp_table.verticalHeader().setSectionResizeMode(ii,
                                                                  QtWidgets.QHeaderView.ResizeToContents)

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

    def comp_table_click(self):
        comps = [x.text() for x in self.comp_table.selectedItems()]
        if not all([x[0] == comps[0][0] for x in comps]):
            QtWidgets.QMessageBox.question(self, 'Message',
                                           'Can\'t mix components with different units',
                                           QtWidgets.QMessageBox.Ok)
            comps = self.dpm.components
            items = self.comp_table.selectedItems()
            # self.comp_table.clearSelection()
            self.comp_table.itemSelectionChanged.disconnect(self.comp_table_click)
            for item in items:
                if item.text() not in comps:
                    # print(item.text())
                    self.comp_table.setCurrentItem(item, QtCore.QItemSelectionModel.Deselect)
            self.comp_table.itemSelectionChanged.connect(self.comp_table_click)
            return
        else:
            self.dpm.components = comps
            self.update_dpm(updated_sites=self.dpm.site_names,
                            updated_comps=self.dpm.components)

    def list_click(self):
        # This doesn't delete comps that are removed from the list
        # I think I need a way to keep track of what is contained in each
        # axis / line so that I can just modify / delete lines as need be,
        # rather than redrawing every time. The only time an axes needs to be
        # redrawn from scratch is if the site changes (although min/max have to
        # be updated if a new comp is added.)
        comps = [x.text() for x in self.comp_list.selectedItems()]
        if not all([x[0] == comps[0][0] for x in comps]):
            QtWidgets.QMessageBox.question(self, 'Message',
                                           'Can\'t mix components with different units',
                                           QtWidgets.QMessageBox.Ok)
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
        # self.comp_list.itemSelectionChanged.connect(self.list_click)
        self.comp_table.itemSelectionChanged.connect(self.comp_table_click)
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
        #  Set up the menu items
        self.actionList_File.triggered.connect(self.WriteList)
        # self.actionData_File.triggered.connect(self.WriteData)
        self.actionWriteWSINV3DMT.triggered.connect(self.write_wsinv3dmt)
        self.actionWriteModEM.triggered.connect(self.write_ModEM)
        self.writeCurrentPlot.triggered.connect(self.write_current_plot)
        self.writeAllPlots.triggered.connect(self.write_all_plots)
        self.actionPhase_Wrap.triggered.connect(self.set_phase_wrap)
        self.regErrors.clicked.connect(self.regulate_errors)
        self.LockAxes.clicked.connect(self.link_axes)
        #  Set up Inversion Type action group
        self.InversionTypeGroup = QtWidgets.QActionGroup(self)
        self.InversionTypeGroup.addAction(self.inv_type1)
        self.InversionTypeGroup.addAction(self.inv_type2)
        self.InversionTypeGroup.addAction(self.inv_type3)
        self.InversionTypeGroup.addAction(self.inv_type4)
        self.InversionTypeGroup.addAction(self.inv_type5)
        self.AzimuthScrollBar.valueChanged.connect(self.azimuth_scroll)
        #  Set up connect to which errors are plotted
        self.actionDataErrors.changed.connect(self.dummy_update_dpm)
        self.actionRawErrors.changed.connect(self.dummy_update_dpm)
        # Super hacky axis limits setters. Fix this at some point
        # self.axmin_hack.editingFinished.connect(self.set_axis_bounds)
        # self.axmax_hack.editingFinished.connect(self.set_axis_bounds)

    def set_phase_wrap(self):
        if self.actionPhase_Wrap.isChecked():
            self.dpm.wrap = 1
        else:
            self.dpm.wrap = 0
        if 'pha' in self.dpm.components[0].lower():
            self.dummy_update_dpm(toggled=True)

    def set_axis_bounds(self):
        try:
            self.dpm.min_ylim = float(self.axmin_hack.text())
            self.dpm.max_ylim = float(self.axmax_hack.text())
            for axnum, ii in enumerate(self.site_names):
                self.dpm.set_bounds(Min=0, Max=4, axnum=axnum)
            self.canvas.draw()
        except ValueError:
            return

    def link_axes(self):
        if self.LockAxes.checkState():
            self.dpm.link_axes_bounds = True
        else:
            self.dpm.link_axes_bounds = False
        self.dpm.redraw_axes()
        self.update_dpm()

    def write_current_plot(self):
        # pass
        filename, ret = self.file_dialog.write_file(ext='.pdf')
        if ret:
            file, extention = os.path.splitext(filename)
            if not extention:
                extention = '.pdf'
            self.dpm.fig.savefig(''.join([file, extention]), dpi=self.fig_dpi)
        # filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Plot',)

    def write_all_plots(self):
        filename, ret = self.file_dialog.write_file(ext='.pdf')
        if ret:
            file, extention = os.path.splitext(filename)
            if not extention:
                extention = '.pdf'
            extention = extention.lower()
            full_file = ''.join([file, extention])
            num_plots = len(self.site_names)
            NS = self.dataset.data.NS
            original_sites = self.site_names
            for ii in range(0, NS, num_plots):
                fig_number = int((ii / num_plots) + 1)
                print('Saving figure {} of {}'.format(fig_number, int(np.ceil(NS / num_plots))))
                sites = self.dataset.data.site_names[ii: ii + num_plots]
                self.dpm.sites = self.dataset.get_sites(site_names=sites, dTypes='all')
                self.dpm.plot_data()
                if extention == '.pdf':
                    if ii == 0:
                        pp = PdfPages(full_file)
                    self.dpm.fig.savefig(pp, format='pdf', dpi=self.fig_dpi)
                else:
                    self.dpm.fig.savefig(''.join([file, str(fig_number), extention]), dpi=self.fig_dpi)
            if extention == '.pdf':
                pp.close()
            print('Done!')
            self.dpm.sites = self.dataset.get_sites(site_names=original_sites, dTypes='all')

    def regulate_errors(self):
        print('Recalculating error maps...')
        fwidth = float(self.width.text())
        mult = float(self.mult.text())
        print([fwidth, mult])
        self.dataset.regulate_errors(multiplier=mult, fwidth=fwidth)
        print('Updating error tree...')
        self.update_error_tree()
        print('Done!')
        self.update_dpm()

    def sort_sites(self, index=None):
        if index is None:
            index = self.sortSites.currentIndex()
        self.dataset.sort_sites(self.sortSites.itemText(index))
        self.siteList.clear()
        self.siteList.addItems(self.dataset.data.site_names)

    def show_map(self):
        # print(self.map_view.map.site_locations['generic'])
        # print(self.map_view.map.generic_sites)
        # print(self.map_view.map.active_sites)
        self.map_view.update_map()
        self.map_view.show()
        # if not self.map['fig']:
            # self.draw_map()
        # self.map['canvas'].show()

    def update_map_data(self):
        self.map_view.periods = self.dataset.data.periods
        self.map_view.PeriodScrollBar.setMaximum(len(self.dataset.data.periods))
        self.map_view.dataset = self.dataset

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
        # ymesh, Ny = utils.generate_mesh(site_locs=locs[:, 1], min_x=self.yMeshMin.value(), DEBUG=False)
        # xmesh, Nx = utils.generate_mesh(site_locs=locs[:, 0], min_x=self.xMeshMin.value(), DEBUG=False)
        # y, x = np.meshgrid(ymesh, xmesh)
        # c = np.ones_like(x)
        # self.map['axis'].pcolor(y, x, c, facecolor='none', edgecolor='k')
        self.map['plots']['all'] = self.map['axis'].plot(locs[:, 1], locs[:, 0], 'k+')
        self.map['plots']['highlight'] = self.map['axis'].plot(clocs[:, 1], clocs[:, 0], 'ro')
        for ii, (ix, iy) in enumerate(clocs):
            self.map['axis'].annotate(self.site_names[ii], xy=(iy, ix))
        self.map['canvas'].draw()

    def dummy_update_dpm(self, toggled=True):
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
        dset = self.currentDataset.itemText(index)
        self.current_dataset = dset
        self.dataset = self.stored_datasets[dset]
        self.site_names = self.shift_site_names(shift=0)
        self.map_view.init_map(dataset=self.dataset,
                               sites=self.dataset.data.site_names,
                               active_sites=self.site_names)
        self.set_data_toggles()
        # Temporarily disconnect the error tree so it doesn't freak out when the data is updated
        # self.error_tree.itemChanged.disconnect()
        self.sort_sites()
        self.update_error_tree()
        self.back_or_forward_button(shift=0)
        self.expand_tree_nodes(to_expand=self.site_names, expand=True)
        self.map_view.update_map()
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
        min_row = np.Infinity
        for site in self.siteList.selectedItems():
            min_row = min(min_row, self.siteList.row(site))
            name = site.text()
            site_names.append(name)
            self.rmSitesList.addItem(self.siteList.takeItem(self.siteList.row(site)))
            self.stored_sites.update({name: {dType: getattr(self.dataset, dType).sites[name]
                                             for dType in self.dataset.data_types
                                             if self.dataset.has_dType(dType)}})
        min_row = max(min_row - 1, 0)
        self.siteList.scrollToItem(self.siteList.item(min_row), hint=1)
        self.dataset.remove_sites(sites=site_names)
        intersect = set(site_names).intersection(set(self.dpm.site_names))
        idx_plotted = list(sorted([self.dataset.data.site_names.index(x) for x in self.dpm.site_names
                                   if x in self.dataset.data.site_names]))
        if intersect:
            max_idx = len(self.dataset.data.site_names) - 1
            # If any of the plotted sites remain
            if idx_plotted:
                # If we are at the end of the list
                if idx_plotted[-1] == max_idx:
                    direction = -1
                    start = idx_plotted[-1]
                    end = 0
                else:
                    direction = 1
                    start = idx_plotted[0]
                    end = max_idx
            # if you removed all the plotted sites
            else:
                start = 0
                end = max_idx
                direction = 1
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
        # Sites were removed, the map should be updated
        self.map_view.map.site_names = self.dataset.site_names
        self.map_view.map.active_sites = self.site_names
        self.map_view.map.set_locations()
        self.map_view.update_map()

    def add_sites(self):
        # This method and the relevent methods in ws.data_structures are
        # not optimized. Adding many sites at once could be slow.
        site_names = []
        min_row = np.Infinity
        for site in self.rmSitesList.selectedItems():
            min_row = min(min_row, self.rmSitesList.row(site))
            name = site.text()
            site_names.append(name)
            self.siteList.addItem(self.rmSitesList.takeItem(self.rmSitesList.row(site)))
            self.dataset.add_site(self.stored_sites[name])
            del self.stored_sites[name]
        self.map_view.map.site_names = self.dataset.site_names
        self.map_view.map.active_sites = self.site_names
        self.map_view.map.set_locations()
        self.map_view.update_map()

    def print_periods(self):
        periods = list(self.dataset.raw_data.narrow_periods.keys())
        periods.sort()
        pretty_periods = [(per, self.dataset.raw_data.narrow_periods[per]) for
                          per in periods]
        print('{:>21} {:>15} {:>15}'.format('Period', 'Log(Period)', 'Perc'))
        print('-' * 56)
        for t in pretty_periods:
            k, v = t
            log_k = np.log10(k)
            if k in utils.truncate(self.dataset.data.periods):
                yn = '*'
            else:
                yn = ''
            if k < 1:
                k = -1 / k
            if self.dataset.freqset:
                cp = utils.closest_periods(self.dataset.freqset, [k])[0]
                idx = self.dataset.freqset.index(cp) + 1
            else:
                idx = ''
            print('{:>2} {:>2} {:15.5} {:15.5} {:15.5} {:<2}'.format(yn, idx, k, log_k, v, yn))
        # for t in self.dataset.data.periods:
        #     if t < 1:
        #         t = -1 / t
        #     print('{}\n'.format(utils.truncate(t)))

    def DEBUG_METHOD(self):
        # print(self.dpm.scale)
        # print(self.current_dataset)
        # print(self.stored_datasets)
        # print(self.dataset)
        # print(self.dataset.azimuth, self.dataset.data.azimuth,
        #       self.dataset.response.azimuth, self.dataset.raw_data.azimuth)
        print((self.dataset.data.sites[
              self.dataset.data.site_names[0]].used_error['ZXYR']))

    def azimuth_scroll(self, val):
        current_val = float(self.azimuthEdit.text())
        if val < self.AzimuthScrollBar.value():
            new_val = (current_val - 1) % 359
        else:
            new_val = (current_val + 1) % 359
        self.AzimuthScrollBar.setValue(new_val)
        self.azimuthEdit.setText(str(new_val))

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

    def write_wsinv3dmt(self):
        self.WriteData(file_format='WSINV3DMT')

    def write_ModEM(self):
        self.WriteData(file_format='ModEM')

    def check_inv_type(self):
        if self.inv_type1.isChecked():
            return 1
        elif self.inv_type2.isChecked():
            return 2
        elif self.inv_type3.isChecked():
            return 3
        elif self.inv_type4.isChecked():
            return 4
        elif self.inv_type5.isChecked():
            return 5
        else:
            return self.dataset.data.inv_type

    def WriteData(self, file_format='WSINV3DMT'):
        self.dataset.data.inv_type = self.check_inv_type()
        keep_going = True
        if self.sortSites.itemText(self.sortSites.currentIndex()) != 'Default':
            reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                   'Site order has changed. Write new list?',
                                                   QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.WriteList()
        while keep_going:
            outfile, ret = MyPopupDialog.get_file(label='Enter data file')
            if not ret or not outfile:
                break
            if ret and outfile:
                if file_format.lower() == 'wsinv3dmt':
                    outfile = utils.check_extention(outfile, expected='data')
                elif file_format.lower() == 'modem':
                    outfile = utils.check_extention(outfile, expected='dat')
                retval = self.dataset.write_data(outfile=outfile, file_format=file_format)
            if retval:
                break
            else:
                reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                       'File already Exists. Overwrite?',
                                                       QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.dataset.write_data(outfile=outfile, overwrite=True, file_format=file_format)
                    break

    def WriteList(self):
        keep_going = True
        while keep_going:
            outfile, ret = MyPopupDialog.get_file(label='Enter list file')
            if not ret or not outfile:
                break
            if ret and outfile:
                outfile = utils.check_extention(outfile, expected='lst')
                retval = self.dataset.write_list(outfile=outfile)
            if retval:
                break
            else:
                reply = QtWidgets.QMessageBox.question(self, 'Message',
                                                       'File already Exists. Overwrite?',
                                                       QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
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
        # self.dpm.fig.canvas.draw()
        self.update_dpm()
        self.expand_tree_nodes(to_expand=self.site_names, expand=True)
        # DEBUG
        # print(self.site_names)
        self.map_view.map.active_sites = self.site_names
        self.map_view.update_map()

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
        self.dpm.which_errors = []
        if self.actionDataErrors.isChecked():
            self.dpm.which_errors.append('data')
        if self.actionRawErrors.isChecked():
            self.dpm.which_errors.append('raw_data')
        if updated_sites is None:
            updated_sites = self.dpm.site_names
        if updated_comps is None:
            updated_comps = self.dpm.components
        dpm_sites = self.dpm.site_names
        updated_sites = list(set(updated_sites).intersection(set(dpm_sites)))  # Sites to be updated
        for site_name in updated_sites:
            ind = next(ii for ii, site in
                       enumerate(self.dpm.site_names) if site == site_name)
            self.dpm.sites['data'][ind] = self.dataset.data.sites[site_name]
            # print('single axes time')
            # t = time.time()
            self.dpm.redraw_single_axis(site_name=site_name, axnum=ind)
            # print(time.time() - t)
        # print(time.time() - t)
        # t = time.time()
        if self.dpm.link_axes_bounds is True:
            self.dpm.link_axes()
        self.dpm.fig.canvas.draw()
        # print(time.time() - t)

    def edit_error_tree(self, column):
        """
        Is triggered when a cell in the error tree is double clicked, and determines
        whether or not that cell is editable. If it is, it allows you to edit the item
        which then calls post_edit_error.
        """
        self.stored_key_presses = QtWidgets.QApplication.keyboardModifiers()
        self.check_key_presses(verbose=True)
        if self.error_tree.selectedIndexes():
            item = self.error_tree.itemFromIndex(self.error_tree.selectedIndexes()[0])
        else:
            item = []
        column = self.error_tree.currentColumn()
        if column >= 2 and item:
            if item.flags():
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
        # QtWidgets.QTreeWidgetItem(['Site', 'Period', 'Component'])
        self.error_tree.setColumnCount(len(header))
        self.error_tree.setHeaderItem(QtWidgets.QTreeWidgetItem(header))
        periods = sorted(self.dataset.data.periods)
        # for site in self.site_names: # Gives the tree for only the plotted sites
        self.tree_dict = {site: [] for site in self.dataset.data.site_names}
        for site in reversed(self.dataset.data.site_names):
            sitenode = QtWidgets.QTreeWidgetItem([site])
            self.tree_dict.update({site: sitenode})  # Stuff the site nodes into here
            self.error_tree.insertTopLevelItem(0, sitenode)
            for ii, p in enumerate(periods):
                # self.error_tree.insertTopLevelItem(1, QtWidgets.QTreeWidgetItem([str(p)]))
                # sitenode.addChild(QtWidgets.QTreeWidgetItem([str(p)]))
                if p < 1:
                    p = round(-1 / p, 1)
                pnode = QtWidgets.QTreeWidgetItem(sitenode, [str(p)])
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
                    print('Adding period {}, freq {}'.format(period, 1 / period))
                    self.update_dpm(updated_sites=self.dpm.site_names,
                                    updated_comp=self.dataset.data.components)
                    self.update_map_data()
        if event.button == 2:
            if israw:
                # print(ind)
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
                print('Removing period {}, freq {}'.format(period, 1 / period))
                self.update_dpm()
                self.update_map_data()
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
        self.model = QtWidgets.QStandardItemModel(self)

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
                    # If there are spaces in a path, make sure they are joined back up
                    if len(parts) > 2:
                        parts[1] = ' '.join(parts[1:])
                    if dType in acceptable:
                        retval[dset].update({dType: parts[1]})
                    elif dType_abr in acceptable:
                        retval[dset].update({dType_abr: parts[1]})
                    else:
                        print('Unrecognized keyword: {}'.format(dType))
            # If a dataset 'COMMON' is present, fill in any blanks in the other datasets
            # with its values
            # if 'common' in [dname.lower() for dname in r'etval.keys()]:
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
            return False
        if 'raw' in retval.keys():
            if 'list' not in retval.keys():
                print('Cannt read raw data with a list file!')
                return False
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
        dialog.message.setText(message)
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
    if not files:
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
    app = QtWidgets.QApplication(sys.argv)  # Starts GUI event loop
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
