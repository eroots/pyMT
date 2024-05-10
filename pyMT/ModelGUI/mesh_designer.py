#!/usr/bin/env python
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
import pyMT.utils as utils
import pyMT.data_structures as DS
import numpy as np
import copy
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtWidgets, QtGui
from pyMT.GUI_common.common_functions import check_key_presses
from pyMT.GUI_common.classes import FileDialog
from pyMT.GUI_common.windows import Modeling1D
from scipy.ndimage import gaussian_filter
from pyMT.e_colours import colourmaps as cm
from pyMT.IO import debug_print
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import sys
import os
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from pyMT import resources


path = os.path.dirname(os.path.realpath(__file__))
with pkg_resources.path(resources, 'mesh_designer.jpg') as p:
    mesh_designer_jpg = str(p)
Ui_MainWindow, QMainWindow = loadUiType(os.path.join(path, 'mesh_designer.ui'))
plt.rcParams['font.size'] = 8


class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, **kwargs):
        NavigationToolbar.__init__(self, canvas, **kwargs)


class model_viewer_2d(QMainWindow, Ui_MainWindow):
    def __init__(self, model, dataset=None, path='./'):
        super(model_viewer_2d, self).__init__()
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
        """
        self.setupUi(self)
        if dataset:
            if type(dataset) is str:
                dataset = DS.Dataset(datafile=dataset)
            self.site_locations = dataset.data.locations
            self.dataset = dataset
        else:
            self.site_locations = []
            self.regenMesh_2.setEnabled(False)
            self.dataset = DS.Dataset()
        self.path = path
        self.orientation = 'xy'
        self.mesh_color = 'k'
        self.x_slice = 0
        self.y_slice = 0
        self.z_slice = 0
        self.rho_cax = [1, 5]
        self.site_marker = '+'
        self.markersize = 5
        self.site_interior = 'w'
        self.colourmap = 'bwr'
        self.mesh_changable = True
        self.modeling_window = None
        self.fig = Figure()
        self.key_presses = {'Control': False, 'Alt': False, 'Shift': False}
        # self.fig = plt.figure()
        self.cid = {'Mesh': []}
        if model:
            if type(model) is str:
                model = DS.Model(model)
        else:
            print('Generating initial model...')
            model = DS.Model(data=dataset.data)
        self.model = copy.deepcopy(model)
        self.dataset.model = self.model
        self.file_dialog = FileDialog(self)
        self.revert_model = copy.deepcopy(model)
        self.plan_view = self.fig.add_subplot(111)
        self.initialized = False
        self.delete_tolerance = 0.5
        self.plan_view.autoscale(1, 'both', 1)
        self.addmpl(self.fig)
        self.redraw_pcolor()
        # self.plan_view.pcolor(self.model.dy, self.model.dx,
        #                      self.model.vals[:, :, self.slice_idx], edgecolors='w', picker=3)
        cursor = Cursor(self.plan_view, useblit=True, color='black', linewidth=2)
        self._widgets = [cursor]
        self.connect_mpl_events()
        self.init_increase_factors()
        self.setup_widgets()
        if self.dataset.data.site_names:
            self.data_info_labels()
        self.init_decade_list()
        self.update_depth_list()
        # self.update_decade_list()
        self.initialized = True

    def init_increase_factors(self):
        self.minDepthFactor.setText(str(self.model.dz[1]))
        self.maxDepthFactor.setText(str(self.model.dz[-1]))
        for ix in range(1, self.increaseFactors.rowCount()):
            for iy in range(0, self.increaseFactors.columnCount()):
                self.increaseFactors.setItem(ix, iy, QtWidgets.QTableWidgetItem(''))
        self.increaseFactors.setItem(0, 0, QtWidgets.QTableWidgetItem(str(round(10**(np.log10(float(self.minDepthFactor.text()))+2)))))
        self.increaseFactors.setItem(1, 0, QtWidgets.QTableWidgetItem(str(round(10**(np.log10(float(self.maxDepthFactor.text())))))))
        self.increaseFactors.setItem(0, 1, QtWidgets.QTableWidgetItem(str(1.1)))
        self.increaseFactors.setItem(1, 1, QtWidgets.QTableWidgetItem(str(1.2)))
        self.increaseFactors.setColumnWidth(0, 75)
        self.increaseFactors.setColumnWidth(1, 50)

    def setup_widgets(self):
        self.minDepth.setText(str(self.model.dz[1]))
        self.maxDepth.setText(str(self.model.dz[-1]))
        self.writeModEM.triggered.connect(self.write_modem)
        self.writeMT3DANI.triggered.connect(self.write_mt3dani)
        self.saveProgress.triggered.connect(self.save_progress)
        self.revertProgress.triggered.connect(self.revert_progress)
        self.regenMesh_2.clicked.connect(self.regenerate_mesh)
        self.addPad.clicked.connect(self.add_pads)
        self.removePad.clicked.connect(self.remove_pads)
        self.minDepth.editingFinished.connect(self.min_depth)
        self.maxDepth.editingFinished.connect(self.max_depth)
        self.genDepthsDecades.clicked.connect(self.generate_depths_decades)
        self.genDepthsFactor.clicked.connect(self.generate_depths_factor)
        self.minDepthFactor.editingFinished.connect(self.min_depth_factor)
        self.maxDepthFactor.editingFinished.connect(self.max_depth_factor)
        self.increaseFactors.itemChanged.connect(self.increase_factor_edit)

        self.minX.setText(str(min(self.model.xCS)))
        self.minY.setText(str(min(self.model.yCS)))
        self.maxX.setText(str(max(self.model.xCS)))
        self.maxY.setText(str(max(self.model.yCS)))
        if self.model.is_half_space():
            self.bgRho.setValue(self.model.vals[0, 0, 0])
        else:
            self.bgRho.setValue(2500)
        self.background_resistivity = float(self.bgRho.value())
        self.bgRho.valueChanged.connect(self.validate_background_rho)
        self.setBackgroundRho.clicked.connect(self.set_background_rho)
        # View page
        self.xSlice.valueChanged.connect(self.validate_x_slice)
        self.ySlice.valueChanged.connect(self.validate_y_slice)
        self.zSlice.valueChanged.connect(self.validate_z_slice)
        #  Set up colour map selections
        self.groupColourmaps = QtWidgets.QActionGroup(self)
        self.groupColourmaps.triggered.connect(self.set_colourmap)
        self.actionJet.setActionGroup(self.groupColourmaps)
        self.actionJet_r.setActionGroup(self.groupColourmaps)
        self.actionJetplus.setActionGroup(self.groupColourmaps)
        self.actionJetplus_r.setActionGroup(self.groupColourmaps)
        self.actionBgy.setActionGroup(self.groupColourmaps)
        self.actionBgy_r.setActionGroup(self.groupColourmaps)
        self.actionBwr.setActionGroup(self.groupColourmaps)
        self.actionBwr_r.setActionGroup(self.groupColourmaps)
        # Mesh line colours
        self.groupMeshColours = QtWidgets.QActionGroup(self)
        self.groupMeshColours.triggered.connect(self.set_mesh_colour)
        self.action_meshWhite.setActionGroup(self.groupMeshColours)
        self.action_meshBlack.setActionGroup(self.groupMeshColours)
        # Set up colour limits
        self.actionRho_cax.triggered.connect(self.set_rho_cax)
        # Smoothing
        self.smoothModel.clicked.connect(self.smooth_model)
        # Display Options
        self.actionLock_Aspect_Ratio.triggered.connect(self.redraw_workaround)
        self.actionAnnotate_Sites.triggered.connect(self.redraw_workaround)
        self.actionMarker_Shape.triggered.connect(self.set_marker_shape)
        self.actionMarker_Size.triggered.connect(self.set_marker_size)
        self.actionMarker_Colour.triggered.connect(self.set_marker_colour)
        #
        self.actionLaunch1DModeling.triggered.connect(self.launch_1d_window)
        #
        self.write_group = QtWidgets.QButtonGroup()
        self.write_group.addButton(self.use_halfspace, 1)
        self.write_group.addButton(self.use_1D_inverse, 2)
        self.write_group.addButton(self.use_1D_synthetic, 3)
        self.use_halfspace.setCheckState(2)
        
    @property
    def cmap(self):
        return cm.get_cmap(self.colourmap)

    @property
    def annotate_sites(self):
        return self.actionAnnotate_Sites.isChecked()
            
    @property
    def lock_aspect_ratio(self):
        return self.actionLock_Aspect_Ratio.isChecked()

    def redraw_workaround(self):
        self.initialized = False
        self.redraw_pcolor()
        self.initialized = True

    def set_marker_size(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Marker Size',
                                                         'Value:',
                                                         self.map.markersize,
                                                         0.1, 20, 2)
        if ok_pressed and d != self.map.markersize:
            self.markersize = d
            self.redraw_pcolor()

    def launch_1d_window(self):
        if not self.modeling_window:
            periods = self.dataset.data.periods
            self.modeling_window = Modeling1D(dataset=self.dataset,
                                              parent=self)
            # self.toggle1DResponse.setEnabled(True)
            # self.toggle1DResponse.clicked.connect(self.plot_1D_response)
            # self.dpm.site1D = self.modeling_window.site
            # self.dpm.sites.update({'1d': [self.dpm.site1D] * len(self.dpm.sites['data'])})
            # self.modeling_window.updated.connect(self.update_1D_response)
        self.modeling_window.show()
        self.use_1D_inverse.setEnabled(True)
        self.use_1D_synthetic.setEnabled(True)

    def set_marker_colour(self):
        colours = ['White', 'Black', 'Blue', 'Red', 'Green', 'Cyan', 'Yellow', 'Magenta']
        codes = {'White': 'w',
                 'Black': 'k',
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
            self.site_interior = code
            self.redraw_pcolor()

    def set_marker_shape(self):
        shapes = ['+', '.', 'o', '*', 'x', '^', 'v']
        d, ok_pressed = QtWidgets.QInputDialog.getItem(self,
                                                       'Marker Shape',
                                                       'Shape:',
                                                       shapes,
                                                       0, False)

        if ok_pressed:
            self.site_marker = d
            self.redraw_pcolor()

    def set_colourmap(self):
        if self.actionJet.isChecked():
            self.colourmap = 'jet'
        if self.actionJet_r.isChecked():
            self.colourmap = 'jet_r'
        if self.actionJetplus.isChecked():
            self.colourmap = 'jet_plus'
        if self.actionJetplus_r.isChecked():
            self.colourmap = 'jet_plus_r'
        if self.actionBgy.isChecked():
            self.colourmap = 'bgy'
        if self.actionBgy_r.isChecked():
            self.colourmap = 'bgy_r'
        if self.actionBwr.isChecked():
            self.colourmap = 'bwr'
        if self.actionBwr_r.isChecked():
            self.colourmap = 'bwr_r'
        self.redraw_pcolor()

    def set_mesh_colour(self):
        if self.action_meshWhite.isChecked():
            self.mesh_color = 'w'
        elif self.action_meshBlack.isChecked():
            self.mesh_color = 'k'
        self.redraw_pcolor()

    def set_rho_cax(self):
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Lower limit',
                                                         'Value:',
                                                         self.rho_cax[0],
                                                         -100, 100, 2)
        if ok_pressed:
            lower = d
        d, ok_pressed = QtWidgets.QInputDialog.getDouble(self,
                                                         'Upper limit',
                                                         'Value:',
                                                         self.rho_cax[1],
                                                         -100, 100, 2)
        if ok_pressed:
            upper = d
        if lower < upper and [lower, upper] != self.rho_cax:
            self.rho_cax = [lower, upper]
            self.redraw_pcolor()
        else:
            print('Invalid colour limits')

    def validate_background_rho(self):
        try:
            self.background_resistivity = float(self.bgRho.value())
        except ValueError:
            self.messages.setText('Background resistivity must be numeric')
            self.bgRho.setText(str(self.background_resistivity))

    def validate_x_slice(self, value):
        if value != self.x_slice:
            if value >= self.model.nx or value < 0:
                self.xSlice.setValue(self.x_slice)
            else:
                self.x_slice = value
            self.redraw_pcolor()

    def validate_y_slice(self, value):
        if value != self.y_slice:
            if value >= self.model.ny or value < 0:
                self.ySlice.setValue(self.y_slice)
            else:
                self.y_slice = value
            self.redraw_pcolor()

    def validate_z_slice(self, value):
        if value != self.z_slice:
            if value >= self.model.nz or value < 0:
                self.zSlice.setValue(self.z_slice)
            else:
                self.z_slice = value
            self.redraw_pcolor()

    def smooth_model(self):
        sigma = [self.sigmaX.value(), self.sigmaY.value(), self.sigmaZ.value()]
        self.model.vals = gaussian_filter(self.model.vals, sigma=sigma)
        self.redraw_pcolor()

    def set_background_rho(self):
        if self.model.is_half_space():
            self.model.vals[:, :, :] = self.background_resistivity
            self.model.background_resistivity = self.background_resistivity
        else:
            reply = QtWidgets.QMessageBox.question(self,
                                                   'Set background',
                                                   'Model is not currently a half-space. Reset it?')
            if reply == QtWidgets.QMessageBox.Yes:
                self.model.background_resistivity = self.background_resistivity
                self.model.vals[:, :, :] = self.background_resistivity
            else:
                return
        self.model.rho_x = self.model.vals
        self.redraw_pcolor()

    def update_dimension_labels(self):
        self.nxLabel.setText(' : '.join(['NX', str(self.model.nx)]))
        self.nyLabel.setText(' : '.join(['NY', str(self.model.ny)]))
        self.nzLabel.setText(' : '.join(['NZ', str(self.model.nz)]))

    def data_info_labels(self):
        locations = self.dataset.data.locations
        avg = 0
        for ii in range(self.dataset.data.NS):
            mask = np.zeros(shape=locations.shape, dtype=int)
            mask[ii, :] = 1
            ma_array = np.ma.masked_array(locations, mask=mask)
            avg += np.min(np.sqrt(np.sum((ma_array - locations[ii, :])**2, axis=1)))
        avg = avg / (self.dataset.data.NS)
        self.spacingLabel.setText('Avg. Station Spacing:{:>8.1f} km'.format(avg / 1000))
        min_avg = 0
        max_avg = 0
        for site in self.dataset.data.site_names:
            rho, depth = utils.compute_bost1D(self.dataset.data.sites[site], filter_width=0.75)[:2]
            min_avg += depth[0]
            max_avg += depth[-1]
        min_avg = (min_avg / self.dataset.data.NS)
        max_avg = (max_avg / self.dataset.data.NS)
        self.minBostLabel.setText('Avg. Min N-B Depth:{:>10.1f} km'.format(min_avg))
        self.maxBostLabel.setText('Avg. Max N-B Depth:{:>9.1f} km'.format(max_avg))

    def min_depth(self):
        try:
            min_depth = float(self.minDepth.text())
        except ValueError:
            min_depth = self.model.dz[1]
            self.minDepth.setText(str(min_depth))
        if min_depth >= float(self.maxDepth.text()):
            self.minDepth.setText(str(0.001))
            self.messages.setText('Minimum depth cannot exceed maximum depth!')
        self.update_decade_list(direction='top')

    def max_depth(self):
        try:
            max_depth = round(float(self.maxDepth.text()))
        except ValueError:
            max_depth = self.model.dz[-1]
            self.maxDepth.setText(str(max_depth))
        if max_depth <= float(self.minDepth.text()):
            self.maxDepth.setText(str(float(self.minDepth.text()) + 1000))
            self.messages.setText('Maximum depth cannot be less minimum depth!')
        self.update_decade_list(direction='bottom')

    def min_depth_factor(self):
        try:
            min_depth = float(self.minDepthFactor.text())
        except ValueError:
            min_depth = self.model.dz[1]
            self.minDepthFactor.setText(str(min_depth))
        if min_depth >= float(self.maxDepthFactor.text()):
            self.minDepthFactor.setText(str(1))
            self.messages.setText('Minimum depth cannot exceed maximum depth!')
        self.update_factor_table()

    def max_depth_factor(self):
        try:
            max_depth = round(float(self.maxDepthFactor.text()))
        except ValueError:
            max_depth = self.model.dz[-1]
            self.maxDepthFactor.setText(str(max_depth))
        if max_depth <= float(self.minDepth.text()):
            self.maxDepthFactor.setText(str(float(self.minDepth.text()) + 1000))
            self.messages.setText('Maximum depth cannot be less minimum depth!')
        self.update_factor_table()

    def update_factor_table(self):
        min_depth = float(self.minDepthFactor.text())
        max_depth = float(self.maxDepthFactor.text())
        for row in range(self.increaseFactors.rowCount()):
            item = self.increaseFactors.item(row, 0)
            try:
                depth = float(item.text())
                if depth <= min_depth:
                    self.increaseFactors.setItem(row, 0, QtWidgets.QTableWidgetItem(str(min_depth + 1)))
                elif depth > max_depth:
                    self.increaseFactors.setItem(row, 0, QtWidgets.QTableWidgetItem(str(max_depth)))
            except (AttributeError, ValueError):
                continue
        # if location == 'bottom':
        #     self.increaseFactors.setItem(self.increaseFactors.rowCount(), 0, QtWidgets.QTableWidgetItem(self.maxDepthFactor.text()))
        # elif location == 'top':
        #     self.increaseFactors.setItem(0, 0, QtWidgets.QTableWidgetItem(self.minDepthFactor.text()))

    def sort_increase_factors(self):
        depths, factors = [], []
        vals = []
        for row in range(self.increaseFactors.rowCount()):
            depths.append(self.increaseFactors.takeItem(row, 0))
            factors.append(self.increaseFactors.takeItem(row, 1))
            try:
                vals.append(float(depths[row].text()))
            except (ValueError, AttributeError):
                num_rows = row
                break
        idx = np.argsort(vals)
        for ii, row in enumerate(idx):
            # print([row, depths[row].text()])
            self.increaseFactors.setItem(ii, 0, depths[row])
            self.increaseFactors.setItem(ii, 1, factors[row])

    def increase_factor_edit(self, item):
        self.increaseFactors.itemChanged.disconnect()
        try:
            row = item.row()
            col = item.column()
            val = float(item.text())
        except ValueError:
            # if col == 0:
                # new_val = round((float(self.maxDepthFactor.text()) + float(self.minDepthFactor.text())) / 2)
                # new_item = QtWidgets.QTableWidgetItem(str(new_val))
            # elif col == 1:
            new_item = QtWidgets.QTableWidgetItem('')
            self.increaseFactors.setItem(row, col, new_item)
            self.messages.setText('Values must be floats.')
            self.increaseFactors.itemChanged.connect(self.increase_factor_edit)
            return
        if col == 0:
            if val <= float(self.minDepthFactor.text()) or val > float(self.maxDepthFactor.text()):
                new_val = round((float(self.maxDepthFactor.text()) + float(self.minDepthFactor.text())) / 2)
                new_item = QtWidgets.QTableWidgetItem(str(new_val))
                self.increaseFactors.setItem(row, 0, new_item)
                self.messages.setText('Factor cutoff depths must be between the min and max depths.')
            elif self.increaseFactors.item(row, 1) is None:
                self.increaseFactors.setItem(row, 1, QtWidgets.QTableWidgetItem('1.1'))
            elif self.increaseFactors.item(row, 1).text() == '':
                self.increaseFactors.setItem(row, 1, QtWidgets.QTableWidgetItem('1.1'))
        elif col == 1:
            if val < 1 or val > 2:
                new_item = QtWidgets.QTableWidgetItem('1.1')
                self.increaseFactors.setItem(row, 1, new_item)
                self.messages.setText('Factors should be between 1 and 2.')
            elif self.increaseFactors.item(row, 0) is None:
                new_val = round((float(self.maxDepthFactor.text()) + float(self.minDepthFactor.text())) / 2)
                new_item = QtWidgets.QTableWidgetItem(str(new_val))
                self.increaseFactors.setItem(row, 0, new_item)
            elif self.increaseFactors.item(row, 0).text() == '':
                new_val = round((float(self.maxDepthFactor.text()) + float(self.minDepthFactor.text())) / 2)
                new_item = QtWidgets.QTableWidgetItem(str(new_val))
                self.increaseFactors.setItem(row, 0, new_item)

        self.sort_increase_factors()
        self.increaseFactors.itemChanged.connect(self.increase_factor_edit)
        

    def update_decade_list(self, direction=None):
        min_depth = float(self.minDepth.text())
        max_depth = float(self.maxDepth.text())
        if direction is None or direction == 'top':
            idx = 0
        else:
            idx = self.zPerDecade.count() - 1
        num_decade = int(np.ceil(np.log10(max_depth)) - np.floor(np.log10(min_depth)))
        while num_decade != self.zPerDecade.count():
            if num_decade < self.zPerDecade.count():
                self.zPerDecade.takeItem(idx)
            elif num_decade > self.zPerDecade.count():
                item = QtWidgets.QListWidgetItem('10')
                item.setFlags(QtCore.Qt.ItemIsEditable |
                              QtCore.Qt.ItemIsSelectable |
                              QtCore.Qt.ItemIsEnabled)
                self.zPerDecade.insertItem(idx, item)
            idx = self.zPerDecade.count() - 1

    def init_decade_list(self):
        min_depth = self.model.zCS[0]
        max_depth = self.model.dz[-1]
        # num_decade = int(np.ceil(np.log10(max_depth)) - np.floor(np.log10(min_dep)))
        idx = 1
        decade = np.log10(min_depth)
        num_per_decade = []
        for iz, z in enumerate(self.model.dz[1:]):
            if np.abs(decade - np.log10(z)) >= 1:
                decade += 1
                num_per_decade.append(idx - 1)
                idx = 1
            else:
                idx += 1
        num_per_decade.append(idx)
        for num in num_per_decade:
            item = QtWidgets.QListWidgetItem(str(num))
            item.setFlags(QtCore.Qt.ItemIsEditable |
                          QtCore.Qt.ItemIsSelectable |
                          QtCore.Qt.ItemIsEnabled)
            self.zPerDecade.insertItem(idx, item)

    def generate_depths_decades(self):
        min_depth = float(self.minDepth.text())
        max_depth = float(self.maxDepth.text())
        depths_per_decade = [float(self.zPerDecade.item(ii).text())
                             for ii in range(self.zPerDecade.count())]
        # if self.zUseDecades.isChecked():
        self.model.generate_zmesh(min_depth=min_depth,
                                  max_depth= max_depth,
                                  decades=depths_per_decade)
        ddz = np.diff(np.diff(self.model.dz[1:]))
        if any(ddz < 0):
            # idx = np.where(ddz)[0][0]
            self.messages.setText('Warning!\n Second derivative of depths is not always positive.')
        else:
            self.messages.setText('Z mesh generation complete.')
        # self.model.generate_zmesh(min_depth, max_depth, depths_per_decade)
        # Update the modeling window with the latest model depths
        if self.modeling_window:
            self.modeling_window.inversion_settings.model_3d = self.model
        self.update_depth_list()
        self.redraw_pcolor()

        # print(ddz)

    def test_table_item(self, item):
        if item is None:
            return True
        elif item.text() == '':
            return True
        else:
            return False

    def generate_depths_factor(self):
        try:
            min_depth = float(self.minDepthFactor.text())
            max_depth = float(self.maxDepthFactor.text())
            depths, factors = [], []
            for row in range(self.increaseFactors.rowCount()):
                depths.append(float(self.increaseFactors.item(row, 0).text()))
                factors.append(float(self.increaseFactors.item(row, 1).text()))
        except (ValueError, AttributeError):
            if len(depths) == len(factors):
                pass
            else:
                self.messages.setText('Make sure the depths / factors are fully specified!')
                return
        factors = [x for _, x in sorted(zip(depths, factors))]
        depths = sorted(depths)
        self.model.generate_zmesh(min_depth=min_depth,
                                  max_depth=max_depth,
                                  increase_factor=[factors, depths])
        ddz = np.diff(np.diff(self.model.dz[1:]))
        if any(ddz < 0):
            # idx = np.where(ddz)[0][0]
            self.messages.setText('Warning!\n Second derivative of depths is not always positive.')
        else:
            self.messages.setText('Z mesh generation complete.')
        self.update_depth_list()
        self.redraw_pcolor()

    def update_depth_list(self):
        self.depthList.setRowCount(self.model.nz)
        self.depthList.setColumnCount(2)
        # self.depthList.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Layer #'))
        self.depthList.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Depth (m)'))
        self.depthList.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem('Thickness (m)'))
        for ii, z in enumerate(self.model.dz):
            self.depthList.setVerticalHeaderItem(ii, QtWidgets.QTableWidgetItem(str(ii)))
            self.depthList.setItem(ii, 0, QtWidgets.QTableWidgetItem(str(round(z,2))))
            if ii > 0:
                self.depthList.setItem(ii, 1, QtWidgets.QTableWidgetItem(str(round(self.model.zCS[ii-1],2))))
        self.depthList.resizeColumnToContents(0)
        self.depthList.resizeColumnToContents(1)

    def add_pads(self):
        # print('Adding pads')
        xmesh = self.model.xCS
        ymesh = self.model.yCS
        if self.padLeft.checkState():
            pad = self.padMult.value() * ymesh[0]
            self.model.dy_insert(0, self.model.dy[0] - pad)
        if self.padRight.checkState():
            pad = self.padMult.value() * ymesh[-1]
            self.model.dy_insert(self.model.ny + 1, pad + self.model.dy[-1])
        if self.padTop.checkState():
            pad = self.padMult.value() * xmesh[-1]
            self.model.dx_insert(self.model.nx + 1, self.model.dx[-1] + pad)
        if self.padBottom.checkState():
            pad = self.padMult.value() * xmesh[0]
            self.model.dx_insert(0, self.model.dx[0] - pad)
        self.redraw_pcolor(x_lim=[self.model.dy[0], self.model.dy[-1]],
                           y_lim=[self.model.dx[0], self.model.dx[-1]])

    def remove_pads(self):
        # print('Removing pads')
        if self.padLeft.checkState():
            self.model.dy_delete(0)
        if self.padRight.checkState():
            self.model.dy_delete(self.model.ny)
        if self.padBottom.checkState():
            self.model.dx_delete(0)
        if self.padTop.checkState():
            self.model.dx_delete(self.model.nx)
        self.redraw_pcolor(x_lim=[self.model.dy[0], self.model.dy[-1]],
                           y_lim=[self.model.dx[0], self.model.dx[-1]])

    def regenerate_mesh(self):
        print('Regenerating mesh')
        self.model.generate_mesh(self.site_locations,
                                 min_x=float(self.minX.text()),
                                 min_y=float(self.minY.text()),
                                 max_x=float(self.maxX.text()),
                                 max_y=float(self.maxY.text()))
        self.redraw_pcolor()

    def revert_progress(self):
        print('Reverting...')
        self.model = copy.deepcopy(self.revert_model)
        self.update_depth_list()
        self.redraw_pcolor()

    def save_progress(self):
        self.revert_model = copy.deepcopy(self.model)

    def write_modem(self):
        self.write_model(write_anisotropic=False)

    def write_mt3dani(self):
        self.write_model(write_anisotropic=True)

    def write_model(self, write_anisotropic=False):
        # model_file, ret = self.file_dialog.write_file(ext='.model', label='Output Model')
        if self.modeling_window:
            if self.use_1D_inverse.checkState():
                if self.modeling_window.inversion_settings.results:
                    model_1D = self.modeling_window.inversion_settings.results['Model']
                    self.model.import_1D(model_1D['Rho'], model_1D['dz'])
                else:
                    ret = QtWidgets.QMessageBox.question(self, 'Write', 'No 1D inverse model is available; Write halfspace?',
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                         QtWidgets.QMessageBox.No)
                    if ret == QtWidgets.QMessageBox.Yes:
                        self.use_halfspace.setCheckState(2)
                    else:
                        return
            elif self.use_1D_synthetic.checkState():
                if self.modeling_window.synthetic_results:
                    model_1D = self.modeling_window.synthetic_results
                    if (model_1D['rho_x'] != model_1D['rho_y']):
                        if  not write_anisotropic:
                            ret = QtWidgets.QMessageBox.question(self, 'Write', 'Synthetic model is anisotropy - use only rho_x?',
                                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                             QtWidgets.QMessageBox.No)
                            if ret == QtWidgets.QMessageBox.Yes:
                                self.model.import_1D(rho_x=model_1D['rho_x'], rho_y=model_1D['rho_y'], dz=model_1D['depth'])
                            else:
                                return
                    else:
                        if model_1D['rho_x']:
                            debug_print(model_1D['rho_x'], 'debug.log')
                            debug_print(model_1D['depth'], 'debug.log')
                            self.model.import_1D(rho_x=model_1D['rho_x'], dz=model_1D['depth'])
                        else:
                            self.background_resistivity = self.modeling_window.synthetic_window.hs
                            self.bgRho.setValue(self.background_resistivity)
                            self.set_background_rho()
                else:
                    ret = QtWidgets.QMessageBox.question(self, 'Write', 'No 1D synthetic model is available; Write halfspace?',
                                                         QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                         QtWidgets.QMessageBox.No)
                    if ret == QtWidgets.QMessageBox.Yes:
                        self.use_halfspace.setCheckState(2)
                    else:
                        return
                    # model_1D = self.modeling_window.synthetic_results
        if self.use_halfspace.checkState():
            self.set_background_rho()
        if write_anisotropic:
            format_string = 'Model Files (*.zani);; All Files (*)'
        else:
            format_string = 'Model Files (*.model *.rho);; All Files (*)'
        model_file, ret = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Model', self.path,
                                                             format_string)[:2]
        if ret:
            cov_file, ret = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Covariance', self.path,
                                                             'Covariance Files (*.cov);; All Files (*)')[:2]
            # cov_file, ret = self.file_dialog.write_file(ext='.cov', label='Output Covariance')
            if write_anisotropic:
                file_format = 'mt3dani'
            else:
                file_format = 'modem'
            self.model.write(model_file, use_anisotropy=write_anisotropic, file_format=file_format)
            if ret:
                self.model.write_covariance(cov_file)

    def connect_mpl_events(self):
        if self.mesh_changable:
            # print('Trying to connect')
            self.cid['Mesh'] = self.canvas.mpl_connect('button_release_event', self.click)
            # self.cid['Mesh'] = self.canvas.mpl_connect('button_press_event', self.click)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)  # Make a canvas
        self.mplvl.addWidget(self.canvas)
        # self.canvas.setParent(self.mplwindow)
        # self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        # self.canvas.setFocus()
        self.toolbar = CustomToolbar(canvas=self.canvas,
                                     parent=self.mplwindow, coordinates=True)
        # Connect check box to instance
        self.canvas.draw()
        self.mplvl.addWidget(self.toolbar)

    def show_legend(self, event):
        """Shows legend for the plots"""
        print('NX: {}, NY: {}, NZ: {}'.format(self.model.nx, self.model.ny, self.model.nz))
        # for pl in [self.x_subplot, self.y_subplot]:
        #     if len(pl.lines) > 0:
        #         pl.legend()
        # plt.draw()

    def clear_xy_subplots(self, event):
        """Clears the subplots."""
        self.model = copy.deepcopy(self.revert_model)
        self.canvas.draw()

    def redraw_pcolor(self, x_lim=None, y_lim=None):
        if self.initialized:
            if not x_lim:
                x_lim = self.plan_view.get_xlim()
            if not y_lim:
                y_lim = self.plan_view.get_ylim()
        self.plan_view.clear()
        # self.plan_view.autoscale(1, 'both', 1)
        self.mesh_plot = self.plan_view.pcolormesh(self.model.dy, self.model.dx,
                                                   np.log10(self.model.vals[:, :, self.z_slice]),
                                                   edgecolors=self.mesh_color, picker=3,
                                                   linewidth=0.1, antialiased=True,
                                                   vmin=self.rho_cax[0], vmax=self.rho_cax[1],
                                                   cmap=self.cmap)
        if np.any(self.site_locations):
            self.location_plot = self.plan_view.plot(self.site_locations[:, 1],
                                                     self.site_locations[:, 0],
                                                     self.site_marker+self.site_interior,
                                                     markersize=self.markersize)
            if self.annotate_sites:
                for ii, (xx, yy) in enumerate(self.site_locations):
                    self.plan_view.annotate(self.dataset.data.site_names[ii], (yy, xx))
        if self.lock_aspect_ratio:
            self.plan_view.set_aspect('equal')
        else:
            self.plan_view.set_aspect('auto')
        if self.initialized:
            self.plan_view.set_xlim(x_lim)
            self.plan_view.set_ylim(y_lim)
        self.canvas.draw()
        self.update_dimension_labels()

    def click(self, event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        #  Don't activate if a toolbar button is being used, or if the click is outside the region
        # if self.toolbar._active:
        if self.toolbar.mode != '':
            return
        if not event.inaxes:
            return
        self.key_presses = check_key_presses(QtWidgets.QApplication.keyboardModifiers())
        # print(key_presses)
        # Get nearest data
        xpos = np.argmin(np.abs(event.xdata - self.model.dy))
        ypos = np.argmin(np.abs(event.ydata - self.model.dx))
        # Check which mouse button:
        if event.button == 1:
            # Plot it
            if self.key_presses['Control']:
                diff = np.abs(event.xdata - self.model.dy[xpos])
                if diff <= self.delete_tolerance * abs(self.model.dy[xpos] - self.model.dy[xpos - 1]):
                    self.model.dy_delete(xpos)
            else:
                if event.xdata > self.model.dy[xpos]:
                    xpos += 1
                self.model.dy_insert(xpos, event.xdata)
        elif event.button == 3:
            if self.key_presses['Control']:
                diff = np.abs(event.ydata - self.model.dx[ypos])
                if diff <= self.delete_tolerance * (self.model.dx[ypos] - self.model.dx[ypos - 1]):
                    self.model.dx_delete(ypos)
            else:
                if event.ydata > self.model.dx[ypos]:
                    ypos += 1
                self.model.dx_insert(ypos, event.ydata)
        self.redraw_pcolor()
        # Show it

    def show_plot(self):
        pass


def main():
    # If a model file is not specified, a uniformly spaced model should be generated based on the data.the
    # i.e., use sites as bounds, and use smallest period as a starting point for cell sizes.
    files = sys.argv[1:]
    for file in files:
        if not utils.check_file(file):
            print('File {} not found.'.format(file))
            return
    files = utils.sort_files(files=files)
    try:
        dataset = DS.Dataset(datafile=files['data'])
    except KeyError:
        print('No data file given. Site locations will not be available.')
        dataset = None
    try:
        model = DS.Model(files['model'])
    except KeyError:
        if dataset is None:
            print('One or more of <model_file> and <data_file> must be given!')
            return
        else:
            print('Generating initial model...')
            model = DS.Model(data=dataset.data)
            # print([model.vals.shape, model.nx, model.ny, model.nz])

    app = QtWidgets.QApplication(sys.argv)
    viewer = model_viewer_2d(model=model, dataset=dataset)
    viewer.setWindowIcon(QtGui.QIcon(mesh_designer_jpg))
    viewer.show()
    ret = app.exec_()
    sys.exit(ret)
    viewer.disconnect_mpl_events()


if __name__ == '__main__':
    main()
