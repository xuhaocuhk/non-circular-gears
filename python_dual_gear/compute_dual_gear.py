from math import pi
import numpy as np


def compute_dual_gear(x: [float], k: int = 1) -> ([float], float, [float]):
    n = len(x)
    target_function = lambda l: sum([xi / (l - xi) for xi in x]) / n - 1.0 / k
    iteration_bound = 50
    bound_left = max(x) * 1.001
    bound_right = max(x) * (k + 1.001)

    assert target_function(bound_left) * target_function(bound_right) < 0
    for i in range(iteration_bound):
        bound_middle = (bound_left + bound_right) / 2
        if target_function(bound_middle) < 0:
            bound_right = bound_middle
        else:
            bound_left = bound_middle
    center_distance = (bound_left + bound_right) / 2

    phi = [0.0] + list(map(lambda x: x * 2 * pi / n, cumulative_sum([xi / (center_distance - xi) for xi in x])))
    normalize_factor = 1.0 / phi[-1] * 2 * pi
    phi = [item * normalize_factor for item in phi]
    t = np.linspace(0, 2 * pi, n + 1)
    phiInv = np.interp(t, phi, t)  # inverse function of phi
    y = [center_distance - xval for xval in np.interp(phiInv, t, x + [x[0]])]
    y = y[:-1]
    phi = [2 * pi - p for p in phi[::-1]]
    return y, center_distance, phi


def cumulative_sum(x: list) -> list:
    length = len(x)
    result = [x[0]]
    for i in range(1, length):
        result.append(result[i - 1] + x[i])
    return result
