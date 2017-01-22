import utils
import numpy as np
from PyQt4 import QtGui, QtCore
import sys
# def enforce(*types):
#     def decorator(f):
#         def new_f(*args, **kwds):
#             # we need to convert args into something mutable
#             newargs = []
#             for (a, t) in zip(args, types):
#                 newargs.append(t(a))  # feel free to have more elaborated convertion
#             return f(*newargs, **kwds)
#         return new_f
#     return decorator


class Example(QtGui.QMainWindow):

    def __init__(self, files):
        super(Example, self).__init__()

        self.initUI(files)

    def initUI(self, files):
        entries = {f: [] for f in files}
        self.gb = QtGui.QGroupBox()
        self.gb.setTitle('Input Files')
        self.setCentralWidget(self.gb)
        gbox = QtGui.QGridLayout()
        for ii, f in enumerate(files):
            entries[f] = QtGui.QLineEdit()
            gbox.addWidget(entries[f], ii, 1)
            gbox.addWidget(QtGui.QLabel(f), ii, 0)
        self.gb.setLayout(gbox)

        # self.setCentralWidget(self.textEdit)
        self.statusBar()
        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        # self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('File dialog')
        self.show()

    def showDialog(self):

        fname = QtGui.QFileDialog.getOpenFileNames(self, 'Open file',
                                                  './')
        print(fname)
        # f = open(fname, 'r')

        # with f:
        #     data = f.read()
        #     self.textEdit.setText(data)


def main():

    app = QtGui.QApplication(sys.argv)
    ex = Example(files=('Raw', 'List'))
    ret = app.exec_()
    sys.exit(ret)


if __name__ == '__main__':
    main()


# def enforce_input(**types):
#     def decorator(func):
#         def new_func(*args, **kwargs):
#             newargs = {}
#             for k, t in types.items():
#                 if isinstance(kwargs[k], t):
#                     newargs.update({k: kwargs[k]})
#                 else:
#                     if t == list:
#                         newargs.update({k: utils.to_list(kwargs[k])})
#                     elif t is np.ndarray:
#                         newargs.update({k: np.array(kwargs[k])})
#                     else:
#                         newargs.update({k: t(kwargs[k])})
#             return func(*args, **newargs)
#         return new_func
#     return decorator


# def enforce_output(*types):
#     def decorator(func):
#         def new_outputs(*args, **kwargs):
#             newouts = []
#             outs = func(*args, **kwargs)
#             for (t, a) in zip(types, outs):
#                 if isinstance(a, t):
#                     newouts.append(a)
#                 else:
#                     if t == list:
#                         newouts.append(utils.to_list(a))
#                     elif t == np.array:
#                         newouts.append(np.array(a))
#                     else:
#                         newouts.append(t(a))
#             return newouts
#         return new_outputs
#     return decorator


# @enforce_input(k2=np.ndarray, k3=list)
# @enforce_output(str, np.ndarray, list)
# def func1(k1=2, k2=2, k3=3):
#     print(type(k1), type(k2), type(k3))
#     return k1, k2, k3


# a, b, c = func1('supalup', k2=[2, 3, 4], k3=np.arange(10))





# class test(object):
#     def __init__(self, val):
#         self.val = val
#         self.val2 = self.val

#     def printval(self):
#         print(self.val)


# class test2(object):
#     def __init__(self, val):
#         self.val2 = val
#         self.test = test(val)

#     def ret_copy(self):
#         retval = self
#         retval.val2 = self.val2 + 1
#         return retval
