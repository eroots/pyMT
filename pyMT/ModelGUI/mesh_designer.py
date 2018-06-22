#!/usr/bin/env python
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
import pyMT.utils as utils
import pyMT.data_structures as WSDS
import numpy as np
import copy
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtWidgets
from pyMT.GUI_common.common_functions import check_key_presses
from pyMT.GUI_common.classes import FileDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import sys
import os

path = os.path.dirname(os.path.realpath(__file__))
Ui_MainWindow, QMainWindow = loadUiType(os.path.join(path, 'mesh_designer.ui'))
plt.rcParams['font.size'] = 8


class model_viewer_2d(QMainWindow, Ui_MainWindow):
    def __init__(self, model, data=None):
        super(model_viewer_2d, self).__init__()
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
        """
        self.setupUi(self)
        if data:
            self.site_locations = data.locations
        else:
            self.site_locations = []
            self.regenMesh_2.setEnabled(False)
        self.orientation = 'xy'
        self.site_marker = 'w+'
        self.mesh_color = 'w'
        self.slice_idx = 0
        self.mesh_changable = True
        self.fig = Figure()
        self.key_presses = {'Control': False, 'Alt': False, 'Shift': False}
        # self.fig = plt.figure()
        self.cid = {'Mesh': []}
        self.model = copy.deepcopy(model)
        self.file_dialog = FileDialog(self)
        self.revert_model = copy.deepcopy(model)
        self.overview = self.fig.add_subplot(111)
        self.delete_tolerance = 0.5
        self.overview.autoscale(1, 'both', 1)
        self.addmpl(self.fig)
        self.redraw_pcolor()
        # self.overview.pcolor(self.model.dy, self.model.dx,
        #                      self.model.vals[:, :, self.slice_idx], edgecolors='w', picker=3)
        cursor = Cursor(self.overview, useblit=True, color='black', linewidth=2)
        self._widgets = [cursor]
        self.connect_mpl_events()
        self.setup_widgets()
        self.update_decade_list()

    def setup_widgets(self):
        self.minDepth.setText(str(self.model.dz[1]))
        self.maxDepth.setText(str(self.model.dz[-1]))
        self.writeModel.triggered.connect(self.write_model)
        self.saveProgress.triggered.connect(self.save_progress)
        self.revertProgress.triggered.connect(self.revert_progress)
        self.regenMesh_2.clicked.connect(self.regenerate_mesh)
        self.addPad.clicked.connect(self.add_pads)
        self.removePad.clicked.connect(self.remove_pads)
        self.minDepth.editingFinished.connect(self.min_depth)
        self.maxDepth.editingFinished.connect(self.max_depth)
        self.genDepths.clicked.connect(self.generate_depths)
        self.minX.setText(str(min(self.model.xCS)))
        self.minY.setText(str(min(self.model.yCS)))
        self.maxX.setText(str(max(self.model.xCS)))
        self.maxY.setText(str(max(self.model.yCS)))

    def update_dimension_labels(self):
        self.nxLabel.setText(' : '.join(['NX', str(self.model.nx)]))
        self.nyLabel.setText(' : '.join(['NY', str(self.model.ny)]))
        self.nzLabel.setText(' : '.join(['NZ', str(self.model.nz)]))

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
            max_depth = float(self.maxDepth.text())
        except ValueError:
            max_depth = self.model.dz[-1]
            self.maxDepth.setText(str(max_depth))
        if max_depth <= float(self.minDepth.text()):
            self.maxDepth.setText(str(float(self.minDepth.text()) + 1000))
            self.messages.setText('Maximum depth cannot be less minimum depth!')
        self.update_decade_list(direction='bottom')

    def update_decade_list(self, direction=None):
        print(self.minDepth.text())
        print(self.maxDepth.text())
        min_depth = float(self.minDepth.text())
        max_depth = float(self.maxDepth.text())
        if direction is None or direction == 'top':
            idx = 0
        else:
            idx = self.zPerDecade.count() - 1
        num_decade = int(np.ceil(np.log10(max_depth)) - np.floor(np.log10(min_depth)))
        while num_decade != self.zPerDecade.count():
            print(idx, num_decade)
            if num_decade < self.zPerDecade.count():
                self.zPerDecade.takeItem(idx)
            elif num_decade > self.zPerDecade.count():
                item = QtWidgets.QListWidgetItem('10')
                item.setFlags(QtCore.Qt.ItemIsEditable |
                              QtCore.Qt.ItemIsSelectable |
                              QtCore.Qt.ItemIsEnabled)
                self.zPerDecade.insertItem(idx, item)
            idx = self.zPerDecade.count() - 1

    def generate_depths(self):
        min_depth = float(self.minDepth.text())
        max_depth = float(self.maxDepth.text())
        depths_per_decade = [float(self.zPerDecade.item(ii).text())
                             for ii in range(self.zPerDecade.count())]
        depths, zCS, ddz = utils.generate_zmesh(min_depth, max_depth, depths_per_decade)
        if any(ddz < 0):
            # idx = np.where(ddz)[0][0]
            self.messages.setText('Warning!\n Second derivative of depths is not always positive.')
        else:
            self.messages.setText('Z mesh generation complete.')
        self.model.generate_zmesh(min_depth, max_depth, depths_per_decade)
        self.redraw_pcolor()

        # print(ddz)

    def add_pads(self):
        print('Adding pads')
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
        self.redraw_pcolor()

    def remove_pads(self):
        print('Removing pads')
        if self.padLeft.checkState():
            self.model.dy_delete(0)
        if self.padRight.checkState():
            self.model.dy_delete(self.model.ny)
        if self.padBottom.checkState():
            self.model.dx_delete(0)
        if self.padTop.checkState():
            self.model.dx_delete(self.model.nx)
        self.redraw_pcolor()

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
        self.redraw_pcolor()

    def save_progress(self):
        self.revert_model = copy.deepcopy(self.model)

    def write_model(self):
        file, ret = self.file_dialog.write_file(ext='.model')
        if ret:
            self.model.write(file)

    def connect_mpl_events(self):
        if self.mesh_changable:
            print('Trying to connect')
            self.cid['Mesh'] = self.canvas.mpl_connect('button_release_event', self.click)

    def addmpl(self, fig):
        self.canvas = FigureCanvas(fig)  # Make a canvas
        self.mplvl.addWidget(self.canvas)
        # self.canvas.setParent(self.mplwindow)
        # self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        # self.canvas.setFocus()
        self.toolbar = NavigationToolbar(canvas=self.canvas,
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

    def redraw_pcolor(self):
        self.overview.clear()
        self.overview.autoscale(1, 'both', 1)
        self.overview.pcolor(self.model.dy, self.model.dx,
                             self.model.vals[:, :, self.slice_idx], edgecolors=self.mesh_color, picker=3)
        if np.any(self.site_locations):
            self.overview.plot(self.site_locations[:, 1],
                               self.site_locations[:, 0],
                               self.site_marker)
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
                print(diff)
                print(self.delete_tolerance * abs(self.model.dy[xpos] - self.model.dy[xpos - 1]))
                if diff <= self.delete_tolerance * abs(self.model.dy[xpos] - self.model.dy[xpos - 1]):
                    self.model.dy_delete(xpos)
            else:
                if event.xdata > self.model.dy[xpos]:
                    xpos += 1
                self.model.dy_insert(xpos, event.xdata)
        elif event.button == 3:
            if self.key_presses['Control']:
                diff = np.abs(event.ydata - self.model.dx[ypos])
                print(diff)
                print(self.delete_tolerance * (self.model.dx[ypos] - self.model.dx[ypos - 1]))
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
        data = WSDS.Data(datafile=files['dat'])
    except KeyError:
        print('No data file given. Site locations will not be available.')
        data = None
    try:
        model = WSDS.Model(files['model'])
    except KeyError:
        if data is None:
            print('One or more of <model_file> and <data_file> must be given!')
            return
        else:
            print('Generating initial model...')
            model = WSDS.Model(data=data)
            print([model.vals.shape, model.nx, model.ny, model.nz])

    app = QtWidgets.QApplication(sys.argv)
    viewer = model_viewer_2d(model=model, data=data)
    viewer.show()
    ret = app.exec_()
    sys.exit(ret)
    viewer.disconnect_mpl_events()


if __name__ == '__main__':
    main()
