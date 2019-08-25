"""
Plotting library using Qt
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import figure_config
import sys
import numpy as np


class Plotter:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = PlotterWindow()
        self.window.resize(*figure_config.figure_size)
        self.window.polygon = Plotter.create_polygon(np.array([(100, 0), (100, 100), (0, 100), (0, 0)]))
        self.window.repaint()
        self.window.grab().save('test.png')
        # self.app.exec_()

    @staticmethod
    def create_polygon(contour: np.ndarray):
        """
        Create a QPolygonF object from the given contour
        :param contour: numpy contour
        :return: Qt Polygon
        """
        points = [QtCore.QPointF(x, y) for x, y in contour]
        print(points)
        return QtGui.QPolygonF(points)

    def __del__(self):
        self.window.close()
        self.app.quit()
        del self.window
        del self.app


class PlotterWindow(QtWidgets.QWidget):
    def __init__(self):
        self.polygon = None
        super().__init__()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(150, 150, 150, 100)))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
        if self.polygon is not None:
            print(f'Drawing polygon {repr(self.polygon.data())}')
            painter.drawPolygon(self.polygon)


if __name__ == '__main__':
    plotter = Plotter()
