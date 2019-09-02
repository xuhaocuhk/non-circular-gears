from PyQt5 import QtCore, QtGui, QtWidgets
import os
import yaml
from typing import Tuple
import numpy as np
import logging
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

Point_T = Tuple[float, float]
Radian_T = float


def load_image(image_file: str) -> QtGui.QImage:
    """
    loads an image from disk file
    :param image_file: the image file to load
    :return: QImage object
    """
    assert os.path.isfile(image_file)
    return QtGui.QImage(image_file)


def align_center(contour: np.ndarray, center: Point_T, image: QtGui.QImage) -> Point_T:
    """
    align the center of a contour to the center in the image
    :param contour: the contour
    :param center: center of the contour
    :param image: the image to align center in
    :return: the corresponding center on the image
    """
    min_x, min_y, max_x, max_y = Polygon(contour).bounds
    x, y = center
    x_ratio = (x - min_x) / (max_x - min_x)
    y_ratio = (y - min_y) / (max_y - min_y)
    w, h = image.width(), image.height()
    return x_ratio * w, y_ratio * h


def rotate_painter(painter: QtGui.QPainter, center: Point_T, rotation_angle: Radian_T) -> QtGui.QPainter:
    """
    rotate a painter counterclockwise with respect to center point
    :param painter: painter to be rotated
    :param center: center of rotation
    :param rotation_angle: angle in radians
    :return: rotated painter
    """
    x, y = center
    painter.translate(-x, -y)
    painter.rotate(np.degrees(rotation_angle))
    painter.translate(x, y)


class Texture:
    def __init__(self, name: str, file: str):
        self.name = name
        self.image = load_image(image_file=file)

    def generate_painter(self, painter_device: QtGui.QPaintDevice, rotation_angle=0.0) -> QtGui.QPainter:
        painter = QtGui.QPainter(painter_device)
        # TODO: scale, align and rotate


def load_textures_from_file(texture_file: str):
    with open(texture_file) as file:
        return [Texture(**data) for data in yaml.safe_load(file)]


predefined_textures = load_textures_from_file(os.path.join(os.path.dirname(__file__), '../textures.yaml'))
