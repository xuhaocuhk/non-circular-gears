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
from typing import Iterable, Tuple, Optional
from plot.qt_textures import predefined_textures
import time

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
        self.window.center_pen = QtGui.QPen(QtGui.QColor(*conf.scatter_point['edge']))
        self.window.center_pen.setWidth(conf.scatter_point['edge_width'])
        self.window.center_brush = QtGui.QBrush(QtGui.QColor(*conf.scatter_point['color']))
        # self.app.exec_()

    @staticmethod
    def create_polygon(contour: np.ndarray) -> QtGui.QPolygonF:
        """
        Create a QPolygonF object from the given contour
        :param contour: numpy contour
        :return: Qt Polygon
        """
        points = [QtCore.QPointF(x, y) for x, y in contour]
        logger.debug(f'Creating polygon with points {points}')
        return QtGui.QPolygonF(points)

    def scaled_polygon(self, contour: np.ndarray) -> QtGui.QPolygonF:
        logger.debug(f'Scaling polygon {contour}')
        return self.create_polygon((contour + self.translation) * self.scaling)

    def draw_contours(self, file_path: str, contours: Iterable[Tuple[str, np.ndarray]],
                      centers: Optional[Iterable[Tuple[float, float]]]):
        """
        draw the given contours and save to an image file
        :param file_path: the path to the image file to save
        :param contours: (color option, contour) of the contours to draw
        :param centers: additional centers to be drawn
        :return: None
        """
        contours = list(contours)
        self.window.polygons = [self.scaled_polygon(contour) for _, contour in contours]
        self.window.pens = [self._get_pen(config) for config, _ in contours]
        self.window.brushes = [self._get_brush(config) for config, _ in contours]
        self.window.setStyleSheet('background-color: white;')
        if centers is not None:
            self.window.centers = [tuple(self.scaling * (np.array(center) + self.translation)) for center in centers]
        else:
            self.window.centers = []
        self._save_canvas(file_path)

    def _save_canvas(self, file_path: str):
        self.window.repaint()
        self.window.grab().save(file_path)

    def _get_pen(self, config: str):
        if 'text_' in config:
            return self.pens['math_drive']
        else:
            return self.pens[config]

    def _get_brush(self, config: str):
        if 'text_' in config:
            return config[5:]
        else:
            return self.brushes[config]

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
        self.centers = []
        self.center_pen = None
        self.center_brush = None
        super().__init__()

    def paintEvent(self, event: QtGui.QPaintEvent):
        assert len(self.pens) == len(self.brushes) and len(self.brushes) == len(self.polygons)
        painter = QtGui.QPainter(self)
        for pen, brush, polygon in zip(self.pens, self.brushes, self.polygons):
            if isinstance(brush, QtGui.QBrush):
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawPolygon(polygon)
            else:
                assert isinstance(brush, str)
                texture = predefined_textures[brush]
                painter = texture.generate_painter(self, np.array(polygon), 0, self.centers[0], pen)

        if self.center_brush is not None and self.center_pen is not None:
            painter.setPen(self.center_pen)
            painter.setBrush(self.center_brush)
            for center in self.centers:
                painter.drawEllipse(QtCore.QPointF(*center), conf.scatter_point['radius'], conf.scatter_point['radius'])


if __name__ == '__main__':
    from math import sin, cos, pi

    test_gear = np.array(
        [(.3 * cos(theta), .4 * sin(theta)) for theta in np.linspace(0, 2 * pi, 1024, endpoint=True)]
    )
    plotter = Plotter()
    plotter.draw_contours('test.png', [('text_awesomeface', test_gear)], [(0, 0)])
