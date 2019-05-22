# TODO
# Camera controls (e.g., Zoom to XY, XZ, YZ views)
# Single rotated slices? Path slice?
# Bring in data and start plotting sites, RMS, tipper, etc...
# 2-D views. Can each tab be run separately? e.g., when in 3-D tab, halt updates
# to 2-D views and vice-versa?
import os
import sys
from copy import deepcopy
from PyQt5 import Qt
import numpy as np
import pyvista as pv
import pyMT.data_structures as WSDS
import e_colours.colourmaps as cm
from pyMT.GUI_common import classes
from PyQt5.uic import loadUiType
from PyQt5 import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


path = os.path.dirname(os.path.realpath(__file__))
# Ui_MainWindow, QMainWindow = loadUiType(os.path.join(path, 'data_plot.ui'))
# UiPopupMain, QPopupWindow = loadUiType(os.path.join(path, 'saveFile.ui'))
UI_ModelWindow, QModelWindow = loadUiType(os.path.join(path, 'model_viewer.ui'))
model_path = 'C:/Users/eroots/phd/ownCloud/data/Regions/MetalEarth/swayze/swz_cull1/norot/mesh/finish/swzall2_lastIter.rho'


def model_to_rectgrid(model):
    grid = pv.RectilinearGrid(np.array(model.dy) / 1000,
                              np.array(model.dx) / 1000,
                              -np.array(model.dz) / 1000)
    vals = np.log10(np.swapaxes(np.flip(model.vals, 2), 0, 1)).flatten(order='F')
    grid.cell_arrays['Resitivity'] = vals
    return grid


class ModelWindow(QModelWindow, UI_ModelWindow):
    def __init__(self, parent=None, show=True):
        super(ModelWindow, self).__init__()
        self.setupUi(self)

        # Make sure the frame fills the tab
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        self.vtk_widget = pv.QtInteractor(self.frame3D)
        vlayout.addWidget(self.vtk_widget)

        self.frame3D.setLayout(vlayout)
        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        self.colourMenu = classes.ColourMenu(self)
        mainMenu.addMenu(self.colourMenu)
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = Qt.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.render_3D)
        meshMenu.addAction(self.add_sphere_action)

        # Add the model
        self.model = WSDS.Model(model_path)
        self.clip_model = WSDS.Model(model_path)
        # self.rect_grid = pv.RectilinearGrid(np.array(self.model.dx),
        #                                     np.array(self.model.dy),
        #                                     np.array(self.model.dz))
        # self.rect_grid.cell_arrays['Resitivity'] = np.log10(self.model.vals.flatten(order='F'))
        self.rect_grid = model_to_rectgrid(self.clip_model)
        self.cmap = cm.jet_plus(64)
        self.clim = [1, 5]
        self.actors = {'X': [], 'Y': [], 'Z': []}
        self.slices = {'X': [], 'Y': [], 'Z': []}
        self.init_widgets()
        # self.clip_volume()
        self.connect_widgets()
        self.render_3D()
        self.view_xy()
        # self.vtk_widget.view_isometric()
        # self.vtk_widget.view_xy()
        if show:
            self.show()

    def init_widgets(self):
        # Set up all the slicing widgets
        self.x_slice = 1
        self.y_slice = 1
        self.z_slice = 1
        self.xSliceEdit.setText('1')
        self.ySliceEdit.setText('1')
        self.zSliceEdit.setText('1')
        self.xSliceLabel.setText('1')
        self.ySliceLabel.setText('1')
        self.zSliceLabel.setText('1')
        self.xSliceSlider.setValue(1)
        self.ySliceSlider.setValue(1)
        self.zSliceSlider.setValue(1)
        self.xSliceSlider.setSingleStep(1)
        self.ySliceSlider.setSingleStep(1)
        self.zSliceSlider.setSingleStep(1)
        self.xSliceSlider.setPageStep(5)
        self.ySliceSlider.setPageStep(5)
        self.zSliceSlider.setPageStep(5)
        self.xSliceSlider.setMinimum(1)
        self.ySliceSlider.setMinimum(1)
        self.zSliceSlider.setMinimum(1)
        self.xSliceSlider.setMaximum(self.model.nx)
        self.ySliceSlider.setMaximum(self.model.ny)
        self.zSliceSlider.setMaximum(self.model.nz - 1)
        self.xSliceCheckbox.setCheckState(2)
        self.ySliceCheckbox.setCheckState(2)
        self.zSliceCheckbox.setCheckState(2)
        # Init clip boxes
        self.x_clip = [0, 0]
        self.y_clip = [0, 0]
        self.z_clip = [0, 0]
        self.x0ClipEdit.setText('0')
        self.y0ClipEdit.setText('0')
        self.z0ClipEdit.setText('0')
        self.x1ClipEdit.setText('0')
        self.y1ClipEdit.setText('0')
        self.z1ClipEdit.setText('0')

    def connect_widgets(self):
        # Slicer widgets
        self.xSliceEdit.editingFinished.connect(self.x_text_change)
        self.ySliceEdit.editingFinished.connect(self.y_text_change)
        self.zSliceEdit.editingFinished.connect(self.z_text_change)
        self.xSliceSlider.valueChanged.connect(self.x_slider_change)
        self.ySliceSlider.valueChanged.connect(self.y_slider_change)
        self.zSliceSlider.valueChanged.connect(self.z_slider_change)
        # Clipping widgets
        self.x0ClipEdit.editingFinished.connect(self.clip_x)
        self.x1ClipEdit.editingFinished.connect(self.clip_x)
        self.y0ClipEdit.editingFinished.connect(self.clip_y)
        self.y1ClipEdit.editingFinished.connect(self.clip_y)
        self.z0ClipEdit.editingFinished.connect(self.clip_z)
        self.z1ClipEdit.editingFinished.connect(self.clip_z)
        self.clipVolume.pressed.connect(self.clip_volume)
        # Slice Checkboxes
        self.xSliceCheckbox.stateChanged.connect(self.x_slice_change)
        self.ySliceCheckbox.stateChanged.connect(self.y_slice_change)
        self.zSliceCheckbox.stateChanged.connect(self.z_slice_change)
        # Colour options
        self.colourMenu.action_group.triggered.connect(self.change_cmap)

    def change_cmap(self):
        self.cmap = cm.get_cmap(self.colourMenu.action_group.checkedAction().text())
        self.render_3D()

    def clip_x(self):
        try:
            x0 = int(self.x0ClipEdit.text())
            x1 = int(self.x1ClipEdit.text())
        except ValueError:
            self.x0ClipEdit.setText(str(self.x_clip[0]))
            self.x1ClipEdit.setText(str(self.x_clip[1]))
        if x0 > np.ceil(self.model.nx / 2):
            self.x0ClipEdit.setText(str(self.x_clip[0]))
        else:
            self.x_clip[0] = x0
        if x1 > np.ceil(self.model.nx / 2):
            self.x1ClipEdit.setText(str(self.x_clip[1]))
        else:
            self.x_clip[1] = x1
        # self.clip_volume()
        # self.render_3D()

    def clip_y(self):
        try:
            y0 = int(self.y0ClipEdit.text())
            y1 = int(self.y1ClipEdit.text())
        except ValueError:
            self.y0ClipEdit.setText(str(self.y_clip[0]))
            self.y1ClipEdit.setText(str(self.y_clip[1]))
        if y0 > np.ceil(self.model.ny / 2):
            self.y0ClipEdit.setText(str(self.y_clip[0]))
        else:
            self.y_clip[0] = y0
        if y1 > np.ceil(self.model.ny / 2):
            self.y1ClipEdit.setText(str(self.y_clip[1]))
        else:
            self.y_clip[1] = y1
        # self.clip_volume()
        # self.render_3D()

    def clip_z(self):
        try:
            z0 = int(self.z0ClipEdit.text())
            z1 = int(self.z1ClipEdit.text())
        except ValueError:
            self.z0ClipEdit.setText(str(self.z_clip[0]))
            self.z1ClipEdit.setText(str(self.z_clip[1]))
        if z0 + z1 > self.model.nz:
            self.z0ClipEdit.setText(str(self.z_clip[0]))
            self.z1ClipEdit.setText(str(self.z_clip[1]))
        else:
            self.z_clip[0] = z0
            self.z_clip[1] = z1
        # self.clip_volume()
        # self.render_3D()

    def x_text_change(self):
        try:
            x = int(self.ySliceEdit.text())
        except ValueError:
            self.xSliceEdit.setText(str(self.x_slice))
            return
        if x < 1 or x > self.model.nx:
            self.xSliceEdit.setText(str(self.x_slice))
        else:
            self.x_slice = x
            self.xSliceSlider.setValue(x)
            self.x_slice_change()

    def y_text_change(self):
        try:
            y = int(self.ySliceEdit.text())
        except ValueError:
            self.ySliceEdit.setText(str(self.y_slice))
            return
        if y < 1 or y > self.model.ny:
            self.ySliceEdit.setText(str(self.y_slice))
        else:
            self.y_slice = y
            self.ySliceSlider.setValue(y)
            self.y_slice_change()

    def z_text_change(self):
        try:
            z = int(self.zSliceEdit.text())
        except ValueError:
            self.zSliceEdit.setText(str(self.z_slice))
            return
        if z < 1 or z > self.model.nz:
            self.zSliceEdit.setText(str(self.z_slice))
        else:
            self.z_slice = z
            self.zSliceSlider.setValue(z)
            self.z_slice_change()

    def x_slider_change(self, value):
        self.xSliceEdit.setText(str(value))
        self.x_slice = value
        self.x_slice_change()

    def y_slider_change(self, value):
        self.ySliceEdit.setText(str(value))
        self.y_slice = value
        self.y_slice_change()

    def z_slider_change(self, value):
        self.zSliceEdit.setText(str(value))
        self.z_slice = value
        self.z_slice_change()

    def x_slice_change(self):
        self.vtk_widget.remove_actor(self.actors['X'])
        if self.xSliceCheckbox.checkState():
            # x_slice = self.clip_model.dx[self.x_slice] / 1000
            # self.slices['X'] = self.rect_grid.slice(normal='y',
            #                                         origin=(0, x_slice, 0))
            self.slices['X'] = self.generate_slice('x')
            self.actors['X'] = self.vtk_widget.add_mesh(self.slices['X'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        self.vtk_widget.update()
        self.show_bounds()

    def y_slice_change(self):
        self.vtk_widget.remove_actor(self.actors['Y'])
        if self.ySliceCheckbox.checkState():
            # y_slice = self.clip_model.dy[self.y_slice] / 1000
            # self.actors['Y'] = self.rect_grid.slice(normal='x',
            #                                         origin=(y_slice, 0, 0))
            self.slices['Y'] = self.generate_slice('y')
            self.actors['Y'] = self.vtk_widget.add_mesh(self.slices['Y'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        self.vtk_widget.update()
        self.show_bounds()

    def z_slice_change(self):
        self.vtk_widget.remove_actor(self.actors['Z'])
        if self.zSliceCheckbox.checkState():
            # z_slice = self.clip_model.dz[self.z_slice] / 1000
            # self.actors['Z'] = self.rect_grid.slice(normal='z', origin=(0, 0, z_slice))
            self.slices['Z'] = self.generate_slice('z')
            self.actors['Z'] = self.vtk_widget.add_mesh(self.slices['Z'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        self.vtk_widget.update()
        self.show_bounds()

    def generate_slice(self, normal='X'):
        if normal == 'x':
            slice_loc = self.clip_model.dx[self.x_slice] / 1000
            gen_slice = self.rect_grid.slice(normal='y',
                                             origin=(0, slice_loc, 0))
        elif normal == 'y':
            slice_loc = self.clip_model.dy[self.y_slice] / 1000
            gen_slice = self.rect_grid.slice(normal='x',
                                             origin=(slice_loc, 0, 0))
        elif normal == 'z':
            slice_loc = -self.clip_model.dz[self.z_slice] / 1000
            gen_slice = self.rect_grid.slice(normal=normal,
                                             origin=(0, 0, slice_loc))
        return gen_slice

    def clip_volume(self):
        # x = [self.model.dx[self.x_clip[0]], self.model.dx[self.model.nx - self.x_clip[1]]]
        # y = [self.model.dy[self.y_clip[0]], self.model.dy[self.model.ny - self.y_clip[1]]]
        # z = [self.model.dz[self.z_clip[0]], self.model.dz[self.model.nz - self.z_clip[1]]]
        # clip = [x[0], x[1], y[0], y[1], z[0], z[1]]
        # self.clipped_volume = self.rect_grid.clip_box(clip)
        self.clip_model = deepcopy(self.model)
        for ix in range(self.x_clip[0]):
            self.clip_model.dx_delete(0)
        for ix in range(self.x_clip[1]):
            self.clip_model.dx_delete(self.clip_model.nx)
        for iy in range(self.y_clip[0]):
            self.clip_model.dy_delete(0)
        for iy in range(self.y_clip[1]):
            self.clip_model.dy_delete(self.clip_model.ny)
        for iz in range(self.z_clip[0]):
            self.clip_model.dz_delete(0)
        for iz in range(self.z_clip[1]):
            self.clip_model.dz_delete(self.clip_model.nz)
        self.rect_grid = model_to_rectgrid(self.clip_model)
        self.xSliceSlider.setMaximum(self.clip_model.nx)
        self.ySliceSlider.setMaximum(self.clip_model.ny)
        self.zSliceSlider.setMaximum(self.clip_model.nz - 1)
        self.render_3D()

    def render_3D(self):
        """ add a sphere to the pyqt frame """
        # sphere = pv.Sphere()
        # self.model = WSDS.Model(model_path)
        # rect_grid = pv.RectilinearGrid(np.array(model.dx), np.array(model.dy), np.array(model.dz))
        # rect_grid.cell_arrays['Resitivity'] = np.log10(model.vals.flatten(order='F'))
        # self.vtk_widget.add_mesh(self.rect_grid)
        self.vtk_widget.clear()
        # This method is a full redraw - but slices should still be taken again even
        # if checks are off otherwise they will not get trimmed
        if self.xSliceCheckbox.checkState():
            self.slices['X'] = self.generate_slice('x')
            self.actors['X'] = self.vtk_widget.add_mesh(self.slices['X'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        if self.ySliceCheckbox.checkState():
            self.slices['Y'] = self.generate_slice('y')
            self.actors['Y'] = self.vtk_widget.add_mesh(self.slices['Y'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        if self.zSliceCheckbox.checkState():
            self.slices['Z'] = self.generate_slice('z')
            self.actors['Z'] = self.vtk_widget.add_mesh(self.slices['Z'],
                                                        cmap=self.cmap,
                                                        clim=self.clim)
        self.vtk_widget.update()
        self.show_bounds()
        self.vtk_widget.reset_camera()

    def show_bounds(self):
        self.vtk_widget.show_grid(bounds=[self.clip_model.dy[0] / 1000, self.clip_model.dy[-1] / 1000,
                                          self.clip_model.dx[0] / 1000, self.clip_model.dx[-1] / 1000,
                                          -self.clip_model.dz[-1] / 1000, self.clip_model.dz[0] / 1000],
                                  xlabel='Easting (km)',
                                  ylabel='Northing (km)',
                                  zlabel='Depth (km)')

    def view_xy(self):
        self.vtk_widget.view_xy()
        # self.vtk_widget.view_vector([0, 0, 1])
        # self.vtk_widget.view_xy()
        self.vtk_widget.update()


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = ModelWindow()
    sys.exit(app.exec_())
    # window.disconnect_mpl_events()
