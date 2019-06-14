from math import pi, isclose
import numpy as np


def compute_dual_gear(x: [float], k: int = 1) -> ([float], float, [float]):
    """
    compute the dual gear with the gear given
    :param x: sample points of drive gear's polar function, theta uniformly in [0, 2*pi)
    :param k: drive gear runs k cycles in one cycle of the driven gear
    :return: y, center_distance, phi
    """
    # calculate the center distance
    n = len(x)
    delta_alpha = 2 * pi / n
    iteration_bound = 50
    bound_left = max(x) * 1.001
    bound_right = max(x) * (k + 1.001)
    target_final_phi = 2 * pi / k
    float_tolerance = 1e-5  # constant

    # target function
    def final_phi_bias(center_distance_estimate):
        """
        difference between the final phi and ideal final phi
        """
        final_phi = sum([delta_alpha * x_i / (center_distance_estimate - x_i) for x_i in x])
        return final_phi - target_final_phi

    # find the center distance
    assert final_phi_bias(bound_left) * final_phi_bias(bound_right) < 0
    for i in range(iteration_bound):
        bound_middle = (bound_left + bound_right) / 2
        if final_phi_bias(bound_middle) < 0:
            bound_right = bound_middle
        else:
            bound_left = bound_middle
    center_distance = (bound_left + bound_right) / 2

    # sum up to get phi
    phi = cumulative_sum([delta_alpha * xi / (center_distance - xi) for xi in x])
    assert isclose(phi[-1], target_final_phi, rel_tol=float_tolerance)
    phi = [0] + phi[:-1]  # convert to our convention

    # calculate the inverse function of phi
    uniform_k_value_points = np.linspace(0, target_final_phi, n + 1, endpoint=True)  # final point for simplicity
    phi_inv = np.interp(uniform_k_value_points, phi + [target_final_phi], np.linspace(0, 2 * pi, n + 1, endpoint=True))
    assert isclose(phi_inv[0], 0, rel_tol=float_tolerance)
    raise Exception("not finished!")

    # duplicate for k times
    pass


def cumulative_sum(x: list) -> list:
    length = len(x)
    result = [x[0]]
    for i in range(1, length):
        result.append(result[i - 1] + x[i])
    return result


if __name__ == '__main__':
    compute_dual_gear([1, 2, 3, 4, 5])
