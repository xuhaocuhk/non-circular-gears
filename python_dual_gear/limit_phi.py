"""
Limiting the maximum phi of a polar contour
"""
import numpy as np
from typing import List
from core.compute_dual_gear import compute_dual_gear
from core.phi_shape_average import differentiate_function

Polar_T = List[float]
Func_T = List[float]


def integrate_circle(function: Func_T) -> float:
    """
    integrate a function from 0 to 2pi
    :param function: the function to be integrated
    :return: integrated value
    """


def limit_phi_derivative(polar: Polar_T, max_phi: float, iteration: int) -> Polar_T:
    """
    Limit the maximum value of phi' in iterations
    :param polar: the contour of input drive
    :param max_phi: the maximum value of phi' that can be taken
    :param iteration: iteration time
    :return: the modified drive polar
    """
    y, center_dist, phi = compute_dual_gear(polar)
    d_phi = differentiate_function(phi)
    for iter in range(iteration):
        d_phi = [min(max_phi, value) for value in d_phi]
