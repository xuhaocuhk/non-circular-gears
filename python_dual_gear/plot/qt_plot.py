"""
Plotting library using Qt
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import figure_config
import sys


class Plotter:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = PlotterWindow()
        self.window.resize(*figure_config.figure_size)
        self.window.show()
        # self.app.exec_()

    def __del__(self):
        self.window.close()
        self.app.quit()
        del self.window
        del self.app


class PlotterWindow(QtWidgets.QWidget):
    def paintEvent(self, event: QtGui.QPaintEvent):
        pass


if __name__ == '__main__':
    print('here')
    input()
