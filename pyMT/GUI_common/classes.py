import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PyQt5 import QtWidgets
import pyMT.utils as utils


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
