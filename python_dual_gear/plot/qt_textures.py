from PyQt5 import QtCore, QtGui, QtWidgets
import os


def load_image(image_file: str) -> QtGui.QImage:
    """
    loads an image from disk file
    :param image_file: the image file to load
    :return: QImage object
    """
    assert os.path.isfile(image_file)
    return QtGui.QImage(image_file)
