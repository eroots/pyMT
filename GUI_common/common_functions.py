from PyQt4 import QtCore, QtGui
import pyMT.utils as utils


def check_key_presses(key_presses, verbose=False):
    msg = ''
    retval = {'Control': False, 'Shift': False, 'Alt': False}
    if key_presses == QtCore.Qt.ShiftModifier:
        retval['Shift'] = True
    elif key_presses == QtCore.Qt.ControlModifier:
        retval['Control'] = True
    elif key_presses == QtCore.Qt.AltModifier:
        retval['Alt'] = True
    elif key_presses == (QtCore.Qt.ControlModifier |
                         QtCore.Qt.ShiftModifier):
        retval['Shift'] = True
        retval['Control'] = True
    elif key_presses == (QtCore.Qt.AltModifier |
                         QtCore.Qt.ShiftModifier):
        retval['Alt'] = True
        retval['Shift'] = True
    elif key_presses == (QtCore.Qt.ControlModifier |
                         QtCore.Qt.AltModifier):
        retval['Control'] = True
        retval['Alt'] = True
    elif key_presses == (QtCore.Qt.ControlModifier |
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


class FileDialog(QtGui.QInputDialog):
    def __init__(self, parent=None):
        super(FileDialog, self).__init__(parent)
        self.accepted = QtGui.QPushButton('OK')
        self.rejected = QtGui.QPushButton('Cancel')
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
                    reply = QtGui.QMessageBox.question(parent, 'Message',
                                                       'File already exists. Overwrite?',
                                                       QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
                    if reply == QtGui.QMessageBox.Yes:
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
                    QtGui.QMessageBox.about(parent, 'Message',
                                            'File not found')
                else:
                    return file, ret
            else:
                return file, 0

