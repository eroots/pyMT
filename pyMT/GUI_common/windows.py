from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import numpy as np
from PyQt5.uic import loadUiType
from PyQt5 import QtWidgets, QtCore
import pyMT.utils as utils
from pyMT.GUI_common.classes import TwoInputDialog
import os
from copy import deepcopy


path = os.path.dirname(os.path.realpath(__file__))
UI_ModelingWindow, QModelingMain = loadUiType(os.path.join(path, '1D_modeling.ui'))
UI_StackedDataWindow, QStackedDataMain = loadUiType(os.path.join(path, 'stacked_data.ui'))
UI_InversionWindow, QInversionWindow = loadUiType(os.path.join(path, '1D_inversion.ui'))

# Allow simple 1D inversions?
# Setting starting model to the 1D (from mesh_designer)
# Set up the inversion window. What kinds of options does this need?
# How feasible is it to borrow / write a 1D inversion scheme?
# Mesh designer still needs to get the common colour menu added

class InversionWindow(QInversionWindow, UI_InversionWindow):
    def __init__(self, parent=None, dataset=None):
        super(InversionWindow, self).__init__()
        self.setupUi(self)
        if dataset:
            dummy_site = dataset.data.sites[dataset.data.site_names[0]]
        self.modeling_window = ModelingMain(dummy_site=dummy_site,
                                             parent=parent)
        self.syntheticDock.setWidget(self.modeling_window)
        self.modeling_window.updated.connect(self.update_data_window)
        self.data_window = StackedDataWindow(dataset=dataset,
                                             parent=parent,
                                             synthetic_response=self.modeling_window.site)
        self.dataDock.setWidget(self.data_window)


    def update_data_window(self):
        self.data_window.synthetic_response = self.modeling_window.site
        if self.data_window.checkResponse.checkState():
            self.data_window.plot_data()

class StackedDataWindow(QStackedDataMain, UI_StackedDataWindow):
    
    def __init__(self, dataset, synthetic_response=None, inverse_response=None, parent=None):
        super(StackedDataWindow, self).__init__()

        self.markers = {'inverted': 'o',
                        'raw':      'o',
                        'modeled': ''}
        self.linestyle = {'inverted': '',
                          'raw':      '',
                          'modeled': '--'}
        self.colours = {'inverted': 'grey',
                        'raw':      'grey',
                        'modeled': 'r'}

        self.setupUi(self)
        self.parent = parent
        self.dataset = dataset
        self.synthetic_response = synthetic_response
        self.figure = Figure()
        self.period_range = np.log10([self.dataset.data.periods[0], self.dataset.data.periods[-1]])
        self.add_mpl(self.figure)
        # self.setup_site_list()
        self.connect_widgets()
        self.plot_data()

    # def setup_list_list(self):
        
    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)  # Make a canvas
        self.mplvl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(canvas=self.canvas,
                                         parent=self.mplwindow, coordinates=True)
        # self.toolbar.setFixedHeight(36)
        # self.toolbar.setIconSize(QtCore.QSize(36, 36))
        # Connect check box to instance
        self.canvas.draw()
        self.mplvl.addWidget(self.toolbar)

    def connect_widgets(self):
        # if self.dataset.raw_data.initialized:
            # self.checkRaw.setCheckState(True)
        # else:
            # self.checkRaw.setCheckState(False)
        # self.checkInverted.setCheckState(True)
        self.checkResponse.setCheckState(False)
        # self.checkRaw.clicked.connect(self.plot_data)
        # self.checkInverted.clicked.connect(self.plot_data)
        self.checkResponse.clicked.connect(self.plot_data)
        self.selectAll.clicked.connect(self.siteList.selectAll)
        self.siteList.addItems(self.dataset.data.site_names)
        self.siteList.selectAll()
        self.siteList.itemSelectionChanged.connect(self.plot_data)
        self.rhoType.currentIndexChanged.connect(self.plot_data)
        self.yLimitsLower.valueChanged.connect(self.set_axis_limits)
        self.yLimitsUpper.valueChanged.connect(self.set_axis_limits)


    def plot_data(self):
        self.figure.clear()
        self.axis = self.figure.add_subplot(111)
        rho_type = self.rhoType.itemText(self.rhoType.currentIndex())
        avg_rho = None
        rho = np.zeros((self.dataset.data.NP, len(self.siteList.selectedItems())))
        for ii, site_item in enumerate(self.siteList.selectedItems()):
            site = site_item.text()
            if rho_type.lower() == 'ssq':
                rho[:, ii] = utils.compute_rho(self.dataset.data.sites[site], calc_comp='ssq', errtype='None')[0]
            elif rho_type.lower() == 'determinant':
                rho[:, ii] = utils.compute_rho(self.dataset.data.sites[site], calc_comp='det', errtype='None')[0]
        avg_rho = np.mean(np.log10(rho), axis=1)
        if avg_rho is not None:
            self.axis.semilogx(self.dataset.data.periods, np.log10(rho),
                                       marker=self.markers['inverted'],
                                       color=self.colours['inverted'],
                                       linestyle=self.linestyle['inverted'], alpha=0.5)
            self.axis.semilogx(self.dataset.data.periods, avg_rho,
                                       marker=self.markers['inverted'],
                                       linestyle=self.linestyle['inverted'],
                                       color='r')
            if self.checkResponse.checkState() and self.synthetic_response:
                rho_1D = utils.compute_rho(self.synthetic_response, calc_comp=rho_type.lower()[:3], errtype='None')[0]
                self.axis.semilogx(self.synthetic_response.periods, np.log10(rho_1D),
                                   marker=self.markers['modeled'],
                                   color=self.colours['modeled'],
                                   linestyle=self.linestyle['modeled'])
            self.set_axis_limits()
        self.canvas.draw()

    def set_axis_limits(self):
        self.axis.set_ylim([self.yLimitsLower.value(), self.yLimitsUpper.value()])
        self.canvas.draw()


class ModelingMain(QModelingMain, UI_ModelingWindow):

    updated = QtCore.pyqtSignal()

    def __init__(self, dummy_site=None, parent=None):
        super(ModelingMain, self).__init__()

        self.setupUi(self)
        self.parent = parent
        self.model_figure = Figure()
        self.default_hs = 1000
        self.thickness = []
        self.hs_thickness = 1000000
        self.default_rho = 1000
        self.default_thickness = 10
        self.depth_lim = [0, 100]
        self.rho_lim = [1, 4]
        if dummy_site:
            self.period_range = np.log10([dummy_site.periods[0], dummy_site.periods[-1]])
        else:
            self.period_range = [-4, 4]
        self.site = deepcopy(dummy_site)
        self.rho_x, self.rho_y = [], []
        self.Z = []
        self.add_mpl(self.model_figure)
        self.setup_model_table()
        self.connect_widgets()
        self.plot_model()
        self.calculate_response()

    @property
    def hs(self):
        return float(self.layerTable.item(0, 1).text())

    def setup_model_table(self):
        header = ['Thickness\n(km)', 'Rho X\n(Ωm)', 'Rho Y\n(Ωm)']
        self.layerTable.setColumnCount(len(header))
        self.layerTable.setRowCount(25)
        for ii, label in enumerate(header):
            self.layerTable.setHorizontalHeaderItem(ii, QtWidgets.QTableWidgetItem(label))
            self.layerTable.setColumnWidth(ii, 60)
        self.layerTable.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem('Half Space'))
        # for ii in range(1, self.layerTable.rowCount()):
        #     self.layerTable.setHorizontalHeaderItem(ii, QtWidgets.QTableWidgetItem('Layer {}'.format(ii)))
        self.layerTable.setItem(0, 1, QtWidgets.QTableWidgetItem(str(self.default_hs)))
        self.layerTable.setItem(0, 0, QtWidgets.QTableWidgetItem(str('')))
        self.layerTable.itemAt(0, 0).setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        for ix in range(1, self.layerTable.rowCount()):
            for iy in range(0, self.layerTable.columnCount()):
                self.layerTable.setItem(ix, iy, QtWidgets.QTableWidgetItem(''))

    def add_mpl(self, fig):
        self.canvas = FigureCanvas(fig)  # Make a canvas
        self.mplvl.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(canvas=self.canvas,
                                         parent=self.mplwindow, coordinates=True)
        # self.toolbar.setFixedHeight(36)
        # self.toolbar.setIconSize(QtCore.QSize(36, 36))
        # Connect check box to instance
        self.canvas.draw()
        self.mplvl.addWidget(self.toolbar)

    def connect_widgets(self):
        self.layerTable.itemChanged.connect(self.model_param_change)
        self.lockXAxis.stateChanged.connect(self.plot_model)
        self.lockYAxis.stateChanged.connect(self.plot_model)
        self.rhoLimits.clicked.connect(self.set_rho_limits)
        self.depthLimits.clicked.connect(self.set_depth_limits)

    def set_rho_limits(self):
        limits, ret = TwoInputDialog.get_inputs(label_1='Lower Limit', label_2='Upper Limit',
                                     initial_1=str(self.rho_lim[0]),
                                     initial_2=str(self.rho_lim[1]),
                                     parent=self,
                                     expected=float)
        if ret:
            if limits[0] < limits[1]:
                self.rho_lim = [float(x) for x in limits]
        self.plot_model()

    def set_depth_limits(self):
        limits, ret = TwoInputDialog.get_inputs(label_1='Lower Limit', label_2='Upper Limit',
                                     initial_1=str(self.depth_lim[0]),
                                     initial_2=str(self.depth_lim[1]),
                                     parent=self,
                                     expected=float)
        if ret:
            if limits[0] < limits[1]:
                self.depth_lim = [float(x) for x in limits]
        self.plot_model()

    def validate_entry(self, item):
        val = item.text()
        try:
            float(val)
            return True
        except ValueError:
            row = item.row()
            col = item.column()
        try:
            if col == 0:
                val = self.thickness[row - 1]
            else:
                val = self.rho[row - 1]
        except IndexError:
            val = ''
        
        self.layerTable.setItem(row, col, QtWidgets.QTableWidgetItem(val))
        return False

    def model_param_change(self, item):
        self.layerTable.itemChanged.disconnect(self.model_param_change)
        ret = self.validate_entry(item)
        if ret:
            row = item.row()
            col = item.column()
            # print(row, col, item.text())
        #     # print(item.text())
            if col == 0:
                # print(self.layerTable.item(row, col + 1).text())
                if self.layerTable.item(row, col + 1).text() == '':
        #             # self.layerTable.setItem(row, col + 1, QtWidgets.QTableWidgetItem(item.text()))
                    self.layerTable.setItem(row, col + 1, QtWidgets.QTableWidgetItem(str(self.default_rho)))
                    self.layerTable.setItem(row, col + 2, QtWidgets.QTableWidgetItem(str(self.default_rho)))
            elif col == 1 and row != 0:
                # print(self.layerTable.item(row, col - 1).text())
                if self.layerTable.item(row, 0).text() == '':
        #             # self.layerTable.setItem(row, col - 1, QtWidgets.QTableWidgetItem(item.text()))
                    self.layerTable.setItem(row, col - 1, QtWidgets.QTableWidgetItem(str(self.default_thickness)))
                if self.layerTable.item(row, col + 1).text() == '':
                    self.layerTable.setItem(row, col + 1, QtWidgets.QTableWidgetItem(item))
            elif col == 2 and row != 0:
                if self.layerTable.item(row, 0).text() == '':
        #             # self.layerTable.setItem(row, col - 1, QtWidgets.QTableWidgetItem(item.text()))
                    self.layerTable.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.default_thickness)))
                if self.layerTable.item(row, col - 1).text() == '':
                    self.layerTable.setItem(row, col - 1, QtWidgets.QTableWidgetItem(item))
            self.update_model()
        self.layerTable.itemChanged.connect(self.model_param_change)

    def update_model(self):
        self.thickness, self.rho_x, self.rho_y = [], [], []
        for layer in range(1, self.layerTable.rowCount()):
            thickness = self.layerTable.item(layer, 0).text()
            rho_x = self.layerTable.item(layer, 1).text()
            rho_y = self.layerTable.item(layer, 2).text()
            if thickness:
                if float(thickness) != 0:
                    self.thickness.append(float(thickness))
                    if rho_x:
                        self.rho_x.append(float(rho_x))
                        self.rho_y.append(float(rho_y))
                    else:
                        self.rho_x.append(self.default_rho)
                        self.rho_y.append(self.default_rho)
        self.plot_model()
        self.calculate_response()

    def plot_model(self):
        self.model_figure.clear()
        self.axis = self.model_figure.add_subplot(111)
        depth = np.cumsum([0] + self.thickness + [100000])
        rho_x = self.rho_x + [self.hs] * 2
        rho_y = self.rho_y + [self.hs] * 2
        self.image = self.axis.step(np.log10(rho_x), depth, linestyle='-')
        self.image = self.axis.step(np.log10(rho_y), depth, linestyle='--')
        # self.axis.set_ylim([0, self.max_plot_depth])
        self.axis.set_ylabel('Depth (km)')
        self.axis.set_xlabel('Rho (Ωm)')
        if self.lockXAxis.checkState():
            self.axis.set_xlim(self.rho_lim)
        if self.lockYAxis.checkState():
            self.axis.set_ylim(self.depth_lim)
        else:
            self.axis.set_ylim([0, 100])
        self.axis.invert_yaxis()
        self.canvas.draw()

    def calculate_response(self):
        for ii, rho in enumerate([self.rho_x, self.rho_y]):
            scale = 1 / (4 * np.pi / 10000000)
            mu = 4 * np.pi * 1e-7
            periods = np.logspace(self.period_range[0], self.period_range[1], 80)
            omega = 2 * np.pi / periods
            # d = np.cumsum(self.thickness + [100000])
            d = np.array(self.thickness + [self.hs_thickness]) * 1000
            r = rho + [self.hs]
            cond = 1 / np.array(r)
            # r = 1 / np.array(r)
            Z = np.zeros(len(periods), dtype=complex)
            rhoa = np.zeros(len(periods))
            phi = np.zeros(len(periods))
            for nfreq, w in enumerate(omega):
                prop_const = np.sqrt(1j*mu*cond[-1] * w)
                C = np.zeros(len(r), dtype=complex)
                C[-1] = 1 / prop_const
                if len(d) > 1:
                    for k in reversed(range(len(r) - 1)):
                        prop_layer = np.sqrt(1j*w*mu*cond[k])
                        k1 = (C[k+1] * prop_layer + np.tanh(prop_layer * d[k]))
                        k2 = ((C[k+1] * prop_layer * np.tanh(prop_layer * d[k])) + 1)
                        C[k] = (1 / prop_layer) * (k1 / k2)
            # #         k2 = np.sqrt(1j*omega[nfreq]*C*mu0/r[k+1]);
            #         g = (g*k2+k1*np.tanh(k1*d[k]))/(k1+g*k2*np.tanh(k1*d[k]));
                Z[nfreq] = 1j * w * mu * C[0]

            rhoa = 1/omega*np.abs(Z)**2
            phi = np.angle(Z, deg=True)

            self.Z = Z
            
            # Update all the data
            if ii == 0:
                self.site.data['ZXYR'] = np.real(Z)
                self.site.data['ZXYI'] = -np.imag(Z)
            else:
                self.site.data['ZYXR'] = -np.real(Z)
                self.site.data['ZYXI'] = np.imag(Z)

        self.site.data['ZXXR'] = 0.00001 * np.real(Z)
        self.site.data['ZXXI'] = 0.00001 * np.imag(Z)
        self.site.data['ZYYR'] = 0.00001 * np.real(Z)
        self.site.data['ZYYI'] = 0.00001 * np.imag(Z)
        self.site.periods = periods
        self.site.data.update({'TZXR': np.zeros(Z.shape)})
        self.site.data.update({'TZXI': np.zeros(Z.shape)})
        self.site.data.update({'TZYR': np.zeros(Z.shape)})
        self.site.data.update({'TZYI': np.zeros(Z.shape)})
        # And all the errors
        self.site.used_error['ZXYR'] = np.real(Z) * 0.05
        self.site.used_error['ZXYI'] = -np.imag(Z) * 0.05
        self.site.used_error['ZYXR'] = -np.real(Z) * 0.05
        self.site.used_error['ZYXI'] = np.imag(Z) * 0.05
        self.site.used_error['ZXXR'] = np.ones(Z.shape)
        self.site.used_error['ZXXI'] = np.ones(Z.shape)
        self.site.used_error['ZYYR'] = np.ones(Z.shape)
        self.site.used_error['ZYYI'] = np.ones(Z.shape)
        self.site.used_error.update({'TZXR': np.ones(Z.shape)})
        self.site.used_error.update({'TZXI': np.ones(Z.shape)})
        self.site.used_error.update({'TZYR': np.ones(Z.shape)})
        self.site.used_error.update({'TZYI': np.ones(Z.shape)})

        self.site.calculate_phase_tensors()
        self.update_parent()

    def update_parent(self):
        # print(self.parent)
        if self.parent:
            # if self.parent.toggle1DResponse.checkState():
            self.updated.emit()

