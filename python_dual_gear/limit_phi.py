"""
Limiting the maximum phi of a polar contour
"""
import numpy as np
from typing import List, Optional, Tuple
from core.compute_dual_gear import compute_dual_gear
from core.phi_shape_average import differentiate_function, rebuild_polar
from shape_processor import toCartesianCoordAsNp
from plot.plot_sampled_function import rotate
from math import pi
import math
from plot.qt_plot import Plotter
import os

Polar_T = List[float]
Func_T = List[float]


def integrate_circle(function: Func_T) -> float:
    """
    integrate a function from 0 to 2pi
    :param function: the function to be integrated
    :return: integrated value
    """
    dist = 2 * pi / len(function)
    return sum((value * dist for value in function))


def limit_phi_derivative(polar: Polar_T, max_phi: float, iteration: int) -> np.ndarray:
    """
    Limit the maximum value of phi' in iterations
    :param polar: the contour of input drive
    :param max_phi: the maximum value of phi' that can be taken
    :param iteration: iteration time
    :return: the modified drive polar
    """
    assert not isinstance(polar, np.ndarray)
    y, center_dist, phi = compute_dual_gear(polar)
    d_phi = differentiate_function(phi)
    for _ in range(iteration):
        d_phi = [min(max_phi, value) for value in d_phi]
        s = integrate_circle(d_phi)
        if math.isclose(s, 2 * pi, abs_tol=1.0e-5):
            break
        d_phi = [value / s * 2 * pi for value in d_phi]
    # rebuild the function with d_phi
    return rebuild_polar(center_dist, d_phi)


def to_cartesian(polar_contour: Polar_T, center: Tuple[float, float], rotation: Optional[float] = None) -> np.ndarray:
    contour = toCartesianCoordAsNp(polar_contour, *center)
    if rotation is not None:
        contour = np.array(rotate(contour, rotation, center))
    return contour


def plot_limited_phi(polar_contour: Polar_T, folder: str, max_phi: float, iteration: int) -> Polar_T:
    """
    plot a contour with derivative of phi limited to the given max phi
    :param polar_contour: the contour to be smoothed
    :param folder: folder to store the plots
    :param max_phi: maximum of phi derivative
    :param iteration: iteration times
    :return: optimized drive
    """
    plotter = Plotter()
    driven, center_dist, phi = compute_dual_gear(polar_contour)
    plotter.draw_contours(os.path.join(folder, 'original_contour.png'),
                          [('math_drive', to_cartesian(polar_contour, (0, 0))),
                           ('math_driven', to_cartesian(driven, (center_dist, 0), phi[0]))],
                          [(0, 0), (center_dist, 0)])
    drive = list(limit_phi_derivative(polar_contour, max_phi, iteration))
    driven, center_dist, phi = compute_dual_gear(drive)
    plotter.draw_contours(os.path.join(folder, f'smoothed_contour_{"%3.4f" % max_phi}.png'),
                          [('math_drive', to_cartesian(drive, (0, 0))),
                           ('math_driven', to_cartesian(driven, (center_dist, 0), phi[0]))],
                          [(0, 0), (center_dist, 0)])
    return drive


if __name__ == '__main__':
    import shape_factory
    from models import find_model_by_name
    import shape_processor
    from shapely.geometry import Point
    from debug_util import MyDebugger

    cart_drive = shape_factory.get_shape_contour(find_model_by_name('starfish'), uniform=True, plots=None)
    drive_polar = shape_processor.toExteriorPolarCoord(Point(0.4402, 0.5096), cart_drive, 1024)
    debugger = MyDebugger(['phi_lim'])
    plot_limited_phi(drive_polar, debugger.get_root_debug_dir_name(), 600, 10)
