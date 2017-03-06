#!/usr/bin/env python
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
import pyMT.utils as utils
import pyMT.data_structures as WSDS
import numpy as np
import copy
from PyQt4.uic import loadUiType
from PyQt4 import QtGui
from pyMT.GUI_common.common_functions import check_key_presses, FileDialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
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
        if data:
            self.site_locations = data.locations
        else:
            self.site_locations = []
            self.regenMesh_2.setEnabled(False)
        self.setupUi(self)
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
        self.minX.setText(str(min(self.model.xCS)))
        self.minY.setText(str(min(self.model.yCS)))
        self.maxX.setText(str(max(self.model.xCS)))
        self.maxY.setText(str(max(self.model.yCS)))

    def setup_widgets(self):
        self.writeModel.triggered.connect(self.write_model)
        self.saveProgress.triggered.connect(self.save_progress)
        self.revertProgress.triggered.connect(self.revert_progress)
        self.regenMesh_2.clicked.connect(self.regenerate_mesh)
        self.addPad.clicked.connect(self.add_pads)
        self.removePad.clicked.connect(self.remove_pads)

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
            self.model.dy_delete(self.model.nx)
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
        file, ret = self.file_dialog.write_file()
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

    def click(self, event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        self.key_presses = check_key_presses(QtGui.QApplication.keyboardModifiers())
        # print(key_presses)
        # Get nearest data
        xpos = np.argmin(np.abs(event.xdata - self.model.dy))
        ypos = np.argmin(np.abs(event.ydata - self.model.dx))
        # Check which mouse button:
        if event.button == 1:
            # Plot it
            if event.xdata > self.model.dy[xpos]:
                xpos += 1
            if self.key_presses['Control']:
                diff = np.abs(event.xdata - self.model.dy[xpos])
                print(diff)
                print(self.delete_tolerance * (self.model.dy[xpos] - self.model.dy[xpos - 1]))
                if diff <= self.delete_tolerance * (self.model.dy[xpos] - self.model.dy[xpos - 1]):
                    self.model.dy_delete(xpos)
            else:
                self.model.dy_insert(xpos, event.xdata)
        elif event.button == 3:
            if event.ydata > self.model.dx[ypos]:
                ypos += 1
            if self.key_presses['Control']:
                diff = np.abs(event.ydata - self.model.dx[ypos])
                print(diff)
                print(self.delete_tolerance * (self.model.dx[ypos] - self.model.dx[ypos - 1]))
                if diff <= self.delete_tolerance * (self.model.dx[ypos] - self.model.dx[ypos - 1]):
                    self.model.dx_delete(ypos)
            else:
                self.model.dx_insert(ypos, event.ydata)
        self.redraw_pcolor()
        # Show it

    def show_plot(self):
        pass


def main():
    app = QtGui.QApplication(sys.argv)
    files = sys.argv[1:]
    for file in files:
        if not utils.check_file(file):
            print('File {} not found.'.format(file))
            return
    files = utils.sort_files(files=files)
    model = WSDS.Model(files['model'])
    data = WSDS.Data(datafile=files['data'])
    viewer = model_viewer_2d(model=model, data=data)
    viewer.show()
    ret = app.exec_()
    sys.exit(ret)
    viewer.disconnect_mpl_events()


if __name__ == '__main__':
    # Build some strange looking data:
    main()

