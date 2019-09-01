from PyQt5 import QtCore, QtGui, QtWidgets
import os
import yaml
from typing import Tuple


def load_image(image_file: str) -> QtGui.QImage:
    """
    loads an image from disk file
    :param image_file: the image file to load
    :return: QImage object
    """
    assert os.path.isfile(image_file)
    return QtGui.QImage(image_file)


class Texture:
    def __init__(self, name: str, file: str):
        self.name = name
        self.image = load_image(image_file=file)

    def generate_painter(self, painter_device: QtGui.QPaintDevice, rotation_angle=0.0) -> QtGui.QPainter:
        painter = QtGui.QPainter(painter_device)


def load_textures_from_file(texture_file: str):
    with open(texture_file) as file:
        return [Texture(**data) for data in yaml.safe_load(file)]


predefined_textures = load_textures_from_file(os.path.join(os.path.dirname(__file__), '../textures.yaml'))
