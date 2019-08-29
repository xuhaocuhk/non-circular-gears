"""
Calculation for shape average by averaging out corresponding phi function and center distance
"""

import numpy as np
from typing import Iterable, Collection
from math import pi
from util_functions import align
from core.compute_dual_gear import compute_dual_gear


def pre_process(function: Collection):
    """
    make sure the phi function is monotonically increasing
    :param function: phi function
    :return: pre-processed function
    """
    func = [-value for value in function]
    func[0] += 2 * pi
    for index in range(1, len(func)):
        while func[index] < func[index - 1]:
            func[index] = func[index] + 2 * pi
    return func


def derivative(function: Collection[float], index: int, start=0, end=2 * pi) -> float:
    """
    get the derivative of the function at given index
    :param function: the given function, uniformly sampled in start to end
    :param index: the index of given function
    :param start: starting point of the function (inclusive)
    :param end: ending point of the function (not inclusive)
    :return: the derivative at index
    """
    prev_index = (index - 1) % len(function)
    next_index = (index + 1) % len(function)
    interval_len = (end - start) / len(function) * 2
    return (function[next_index] - function[prev_index]) % (2 * pi) / interval_len


def differentiate_function(function: Collection[float], start=0, end=2 * pi) -> np.ndarray:
    """
    get the derivative of the whole function
    :param function: the given function, uniformly sampled in start to end
    :param start: starting point of the function (inclusive)
    :param end: ending point of the function (not inclusive)
    :return: the derivative of the function
    """
    return np.array([derivative(function, index, start, end) for index in range(len(function))])


def rebuild_polar(center_distance: float, d_phi: Iterable[float]) -> np.ndarray:
    """
    rebuild the polar coordinates from given center distance
    :param center_distance: the center distance
    :param d_phi: derivative of phi
    :return: rebuilt polar function
    """
    return np.array([center_distance * phi_prime / (1 + phi_prime) for phi_prime in d_phi])


def shape_average(polygon_a: Iterable[float], polygon_b: Iterable[float]) -> np.ndarray:
    """
    calculate the average shape of two polygons
    :param polygon_a: polar coordinates for polygon a
    :param polygon_b: polar coordinates for polygon b
    :return: the average polygon
    """
    # retrieve the phi functions and center distances
    _, center_distance_a, phi_a = compute_dual_gear(list(polygon_a))
    _, center_distance_b, phi_b = compute_dual_gear(list(polygon_b))  # compute_dual_gear does not support ndarray yet
    phi_a = pre_process(phi_a)
    phi_b = pre_process(phi_b)
    dphi_a = differentiate_function(phi_a)
    dphi_b = differentiate_function(phi_b)

    # align the derivatives
    offset = align(list(dphi_a), list(dphi_b))
    average_dphi = [(dphi_a[index - offset] + dphi_b[index]) / 2 for index in range(len(dphi_a))]
    center_distance = (center_distance_a + center_distance_b) / 2
    return rebuild_polar(center_distance, np.array(average_dphi))


if __name__ == '__main__':
    from debug_util import MyDebugger
    from shape_processor import toCartesianCoordAsNp, toExteriorPolarCoord
    from shapely.geometry import Point
    from plot.qt_plot import Plotter
    from math import sin, cos

    debugger = MyDebugger('average_test')
    ellipse = toExteriorPolarCoord(Point(0, 0), np.array(
        [(0.3 * cos(theta), 0.2 * sin(theta)) for theta in np.linspace(0, 2 * pi, 1024, endpoint=False)]), 1024)
    ellipse_copy = toExteriorPolarCoord(Point(0, 0), np.array(
        [(0.2 * cos(theta), 0.3 * sin(theta)) for theta in np.linspace(0, 2 * pi, 1024, endpoint=False)]), 1024)
    plotter = Plotter()

    # save the figures
    plotter.draw_contours(debugger.file_path('a.png'), [('math_drive', toCartesianCoordAsNp(ellipse, 0, 0))], None)
    plotter.draw_contours(debugger.file_path('b.png'), [('math_driven', toCartesianCoordAsNp(ellipse_copy, 0, 0))],
                          None)
    print(shape_average(ellipse, ellipse_copy))
    plotter.draw_contours(debugger.file_path('result.png'),
                          [('math_drive', toCartesianCoordAsNp(shape_average(ellipse, ellipse_copy), 0, 0))], None)
