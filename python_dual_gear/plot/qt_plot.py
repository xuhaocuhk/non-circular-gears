"""
Plotting library using Qt
"""
from PyQt5 import QtCore, QtGui, QtWidgets
import figure_config
import sys
import numpy as np
import figure_config as conf
import logging
import itertools

logger = logging.getLogger(__name__)


class Plotter:
    types = ('input', 'math', 'carve')
    pens = {
        type + '_' + driving: QtGui.QPen(QtGui.QColor(*getattr(conf, type + '_shapes')[driving + '_edge']))
        for type, driving in itertools.product(types, ('drive', 'driven'))
    }
    brushes = {
        type + '_' + driving: QtGui.QBrush(QtGui.QColor(*getattr(conf, type + '_shapes')[driving + '_face']))
        for type, driving in itertools.product(types, ('drive', 'driven'))
    }
    for pen in pens.values():
        pen.setWidth(conf.edge_width)

    def __init__(self, translation=conf.figure_translation, scaling=conf.figure_scale):
        self.app = QtWidgets.QApplication(sys.argv)
        self.translation = translation
        self.scaling = scaling
        self.window = PlotterWindow()
        self.window.resize(*figure_config.figure_size)
        self.window.polygons = [self.scaled_polygon(np.array([(0.5, -0.5), (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)]))]
        self.window.pens = [self.pens['input_drive']]
        self.window.brushes = [self.brushes['input_drive']]
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
        logger.debug(f'Creating polygon with points {points}')
        print(points)
        return QtGui.QPolygonF(points)

    def scaled_polygon(self, contour: np.ndarray):
        logger.debug(f'Scaling polygon {contour}')
        return self.create_polygon((contour + self.translation) * self.scaling)

    def __del__(self):
        self.window.close()
        self.app.quit()
        del self.window
        del self.app


class PlotterWindow(QtWidgets.QWidget):
    def __init__(self):
        self.polygons = []
        self.pens = []
        self.brushes = []
        super().__init__()

    def paintEvent(self, event: QtGui.QPaintEvent):
        assert len(self.pens) == len(self.brushes) and len(self.brushes) == len(self.polygons)
        painter = QtGui.QPainter(self)
        for pen, brush, polygon in zip(self.pens, self.brushes, self.polygons):
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawPolygon(polygon)


if __name__ == '__main__':
    print(Plotter.pens, Plotter.brushes)
    plotter = Plotter()
