import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PyQt5 import QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot, Qt
import pyMT.utils as utils
from pyMT.data_structures import RawData, Model, Data
import re
import os
import time

path = os.path.dirname(os.path.realpath(__file__))
UiPopupMain, QPopupWindow = loadUiType(os.path.join(path, 'saveFile.ui'))


class DraggablePoint:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None  # only one can be animated at a time

    def __init__(self, parent, x=0.1, y=0.1, size=(0.1, 0.1), **kwargs):
        self.fc = kwargs.get('fc', 'r')
        self.alpha = kwargs.get('alpha', 0.5)
        self.mec = kwargs.get('mec', 'k')

        self.parent = parent
        self.point = patches.Ellipse((x, y),
                                     size[0],
                                     size[1],
                                     fc=self.fc,
                                     alpha=self.alpha,
                                     edgecolor=self.mec)
        self.x = x
        self.y = y
        parent.fig.axes[0].add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()

        if self.parent.list_points:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]

            self.line = Line2D(line_x, line_y, color=self.fc, alpha=self.alpha)
            parent.fig.axes[0].add_line(self.line)

    def connect(self):
        '''connect to all the events we need'''

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event',
                                                             self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event',
                                                               self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event',
                                                              self.on_motion)

    def on_press(self, event):

        if event.inaxes != self.point.axes:
            return
        if DraggablePoint.lock is not None:
            return
        contains, attrd = self.point.contains(event)
        if not contains:
            return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        if self == self.parent.list_points[1]:
            self.line.set_animated(True)
        else:
            self.parent.list_points[1].line.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):

        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes:
            return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0] + dx, self.point.center[1] + dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        if self == self.parent.list_points[1]:
            axes.draw_artist(self.line)
        else:
            self.parent.list_points[1].line.set_animated(True)
            axes.draw_artist(self.parent.list_points[1].line)

        self.x = self.point.center[0]
        self.y = self.point.center[1]

        if self == self.parent.list_points[1]:
            line_x = [self.parent.list_points[0].x, self.x]
            line_y = [self.parent.list_points[0].y, self.y]
            self.line.set_data(line_x, line_y)
        else:
            line_x = [self.x, self.parent.list_points[1].x]
            line_y = [self.y, self.parent.list_points[1].y]

            self.parent.list_points[1].line.set_data(line_x, line_y)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        '''on release we reset the press data'''
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        if self == self.parent.list_points[1]:
            self.line.set_animated(False)
        else:
            self.parent.list_points[1].line.set_animated(False)

        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

        self.x = self.point.center[0]
        self.y = self.point.center[1]

    def disconnect(self):
        '''disconnect all the stored connection ids'''

        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


class FileDialog(QtWidgets.QInputDialog):
    def __init__(self, parent=None):
        super(FileDialog, self).__init__(parent)
        self.accepted = QtWidgets.QPushButton('OK')
        self.rejected = QtWidgets.QPushButton('Cancel')
        # self.accepted.connect(self.accept)
        # self.rejected.connect(self.reject)

    @staticmethod
    def write_file(parent=None, default=None, label='Input', ext=None):
        if ext is None:
            ext = ''
        while True:
            dialog = FileDialog(parent)
            dialog.setTextValue(default)
            dialog.setLabelText(label)
            ret = dialog.exec_()
            file = dialog.textValue()
            if ret and file:
                if utils.check_file(file) or utils.check_file(''.join([file, ext])):
                    reply = QtWidgets.QMessageBox.question(parent, 'Message',
                                                           'File already exists. Overwrite?',
                                                           QtWidgets.QMessageBox.Yes,
                                                           QtWidgets.QMessageBox.No)
                    if reply == QtWidgets.QMessageBox.Yes:
                        return file, ret
                else:
                    return file, ret
            else:
                return file, 0

    @staticmethod
    def read_file(parent=None, default=None, label='Output'):
        while True:
            dialog = FileDialog(parent)
            dialog.setTextValue(default)
            dialog.setLabelText(label)
            ret = dialog.exec_()
            file = dialog.textValue()
            if ret and file:
                if not utils.check_file(file):
                    QtWidgets.QMessageBox.about(parent, 'Message',
                                                'File not found')
                else:
                    return file, ret
            else:
                return file, 0


class TwoInputDialog(QtWidgets.QDialog):
    def __init__(self, label_1, label_2, initial_1=None, initial_2=None, parent=None):
        super(QtWidgets.QDialog, self).__init__(parent)
        label1 = QtWidgets.QLabel(label_1, self)
        label2 = QtWidgets.QLabel(label_2, self)
        hbox = QtWidgets.QHBoxLayout()
        self.setLayout(hbox)
        hbox.addWidget(label1)
        hbox.addWidget(label2)
        self.value_1 = initial_1
        self.value_2 = initial_2
        self.line_edit1 = QtWidgets.QLineEdit(initial_1)
        self.line_edit2 = QtWidgets.QLineEdit(initial_2)
        hbox.addWidget(self.line_edit1)
        hbox.addWidget(self.line_edit2)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                                             self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        hbox.addWidget(buttons)
        # self.cancel.slicked.connect(self.cancel_button)

    def inputs(self):
        return self.line_edit1.text(), self.line_edit2.text()

    @staticmethod
    def get_inputs(label_1, label_2, initial_1=None, initial_2=None, parent=None, expected=None):
        dialog = TwoInputDialog(label_1, label_2, initial_1, initial_2, parent)
        result = dialog.exec_()
        inputs = dialog.inputs()
        if expected:
            try:
                result = [expected(x) for x in inputs]
            except ValueError:
                QtWidgets.QMessageBox.warning(self, '...', 'Inputs not valid. Must be {}'.format(expected))
                result = False
        return inputs, result


class CoordinateMenu(QtWidgets.QMenu):
    def __init__(self, parent=None):
        super(QtWidgets.QMenu, self).__init__(parent)
        self.default_units = 'km'
        self.default_system = 'Local'
        self.all_systems = {x: None for x in ('Local',
                                              'Lambert',
                                              'UTM',
                                              'LatLong')}
        self.all_units = {x: None for x in ('km',
                                            'm')}
        self.setTitle('Coordinate Options')
        self.unitMenu = self.addMenu('Units')
        self.unitGroup = QtWidgets.QActionGroup(self)
        self.systemMenu = self.addMenu('Coordinate System')
        self.systemGroup = QtWidgets.QActionGroup(self)
        for item in self.all_systems.keys():
            self.all_systems[item] = QtWidgets.QAction(item, parent, checkable=True)
            self.systemMenu.addAction(self.all_systems[item])
            self.systemGroup.addAction(self.all_systems[item])
        for item in self.all_units.keys():
            self.all_units[item] = QtWidgets.QAction(item, parent, checkable=True)
            self.unitMenu.addAction(self.all_units[item])
            self.unitGroup.addAction(self.all_units[item])
        self.unitGroup.setExclusive(True)
        self.systemGroup.setExclusive(True)
        self.all_systems[self.default_system].setChecked(True)
        self.all_units[self.default_units].setChecked(True)
        

class ColourMenu(QtWidgets.QMenu):
    def __init__(self, parent=None):
        super(QtWidgets.QMenu, self).__init__(parent)
        self.default_cmap = 'turbo'
        self.all_maps = {x: None for x in ('bgy', #'bgy_r',
                                           'viridis', #'viridis_r',
                                           'jet', #'jet_r',
                                           'jet_plus',# 'jet_plus_r',
                                           'bwr', #'bwr_r',
                                           'greys', #'greys_r',
                                           'turbo', #'turbo_r', 
                                           'turbo_capped', #'turbo_capped_r', 
                                           'turbo_mod',
                                           'twilight', 
                                           'twilight_shifted',
                                           'colorwheel',
                                           'hot', #'hot_r',
                                           'viridis')} #'viridis_r')}
        self.action_group = QtWidgets.QActionGroup(self)
        self.setTitle('Colour Options')
        self.map = self.addMenu('Colour Map')
        self.limits = self.addAction('Colour Limits')
        self.lut = self.addAction('# Colour Intervals')

        # Add all the colour maps
        self.map.invert_cmap = QtWidgets.QAction('Invert Colourmap', parent, checkable=True)
        self.map.addAction(self.map.invert_cmap)
        self.map.invert_cmap.setChecked(True)
        self.map.addSeparator()
        for item in self.all_maps.keys():
            self.all_maps[item] = QtWidgets.QAction(item, parent, checkable=True)
            self.map.addAction(self.all_maps[item])
            self.action_group.addAction(self.all_maps[item])
            # self.map.addMenu(item)
        self.action_group.setExclusive(True)
        self.all_maps[self.default_cmap].setChecked(True)

    def set_clim(self, initial_1='1', initial_2='5'):
        inputs, ret = TwoInputDialog.get_inputs(label_1='Lower Limit', label_2='Upper Limit',
                                                initial_1=initial_1, initial_2=initial_2, parent=self)
        return inputs, ret

    def set_lut(self, initial='32'):
        inputs, ret = QtWidgets.QInputDialog.getInt(self, 'LUT', 'Number of Colour Intervals:', 
                                                    initial, 1, 1024, 2)
        return inputs, ret


class IOWorker(QObject):
    finished = pyqtSignal()
    # raw_load = pyqtSignal()
    # raw_init = pyqtSignal()
    # data_load = pyqtSignal()
    # data_init = pyqtSignal()
    # response_load = pyqtSignal()
    # response_init = pyqtSignal()
    # model_load = pyqtSignal()
    # model_init = pyqtSignal()
    input_load = pyqtSignal(str)
    input_init = pyqtSignal(str)
    counter = pyqtSignal()

    def __init__(self, file_dict):
        super().__init__()
        self.file_dict = file_dict
        self.ret_val = {}

    @pyqtSlot()
    def proc_counter(self):  # A slot takes no params

        for key, val in self.file_dict.items():
            if key == 'list':
                self.ret_val.update({'raw_data': RawData(listfile=self.file_dict['list'], progress_bar=self)})
            elif key == 'data':
                self.ret_val.update({'data': Data(datafile=self.file_dict['data'], progress_bar=self)})
            elif key == 'model':
                self.ret_val.update({'model': Model(modelfile=self.file_dict['model'], progress_bar=self)})
        self.finished.emit()


class PopUpProgress(QtWidgets.QWidget):

    finished = pyqtSignal()

    def __init__(self, file_dict):
        super().__init__()
        self.pbar = QtWidgets.QProgressBar(self)
        self.pbar.setGeometry(30, 40, 500, 75)
        self.pbar.setTextVisible(True)
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.pbar)
        self.setLayout(self.layout)
        self.setGeometry(300, 300, 550, 100)
        self.setWindowTitle('Progress Bar')
        self.file_dict = file_dict
        self.ret_val = 'Still Nothing'
        self.counter = 0
        self.set_maximum()
        # self.show()

        self.obj = IOWorker(file_dict=file_dict)
        self.thread = QThread()
        self.obj.counter.connect(self.on_count_changed)
        self.obj.moveToThread(self.thread)
        self.obj.finished.connect(self.quit_thread)
        self.obj.input_load.connect(self.input_load)
        self.obj.input_init.connect(self.input_init)
        self.obj.finished.connect(self.close_bar)  # To hide the progress bar after the progress is completed
        self.thread.started.connect(self.obj.proc_counter)
        # self.thread.start()  # This was moved to start_progress

    def input_init(self, input):
        if input.lower() == 'raw':
            self.currentMessage = 'Initializing RawData structure...'
        elif input.lower() == 'model':
            self.currentMessage = 'Initializing Model structure...'
        elif input.lower() == 'data':
            self.currentMessage = 'Initializing Data structure...'

    def input_load(self, input):
        if input.lower() == 'raw':
            self.currentMessage = 'Loading raw data files...'
        elif input.lower() == 'model':
            self.currentMessage = 'Loading inversion model...'
        elif input.lower() == 'data':
            self.currentMessage = 'Loading inversion data...'

    def set_maximum(self):
        maximum = 0
        for key, val in self.file_dict.items():
            if key == 'list':
                with open(self.file_dict['list'], 'r') as f:
                    lines = f.readlines()
                    maximum += len(lines)*2 - 2
            elif key == 'data':
                NS, NP = read_data_header(val)
                maximum += NS
            elif key == 'model':
                maximum += 15

        self.pbar.setMaximum(maximum)

    def start_progress(self):  # To restart the progress every time
        self.show()
        self.thread.start()

    def on_count_changed(self):
        self.counter += 1
        self.pbar.setValue(self.counter)
        self.pbar.setFormat(self.currentMessage)
        self.pbar.setAlignment(Qt.AlignCenter)

    def close_bar(self):
        self.ret_val = self.obj.ret_val
        self.counter = 0
        self.hide()

    def quit_thread(self):
        self.ret_val = self.obj.ret_val
        self.thread.quit()
        self.finished.emit()
        self.close()


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
                print('Cannot read raw data with a list file!')
                return False
        return retval


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