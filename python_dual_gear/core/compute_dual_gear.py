from math import pi, isclose, cos, sin
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from debug_util import MyDebugger
import os


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
    assert isclose(bound_left, bound_right, rel_tol=float_tolerance)

    # sum up to get phi
    phi = cumulative_sum([delta_alpha * xi / (center_distance - xi) for xi in x])
    assert isclose(phi[-1], target_final_phi, rel_tol=float_tolerance)
    phi = [0] + phi[:-1]  # convert to our convention

    # calculate the inverse function of phi
    uniform_k_value_points = np.linspace(0, target_final_phi, n + 1, endpoint=True)  # final point for simplicity
    phi_inv = np.interp(uniform_k_value_points, phi + [target_final_phi], np.linspace(0, 2 * pi, n + 1, endpoint=True))
    assert isclose(phi_inv[0], 0, rel_tol=float_tolerance)

    # calculate the driven gear curve
    phi_inv = phi_inv[::-1]  # flip phi_inv
    y = [center_distance - x_len for x_len in np.interp(phi_inv, np.linspace(0, 2 * pi, n + 1, True), x + [x[0]])]
    y = y[:-1]  # drop the last one
    assert len(y) == len(phi)

    # duplicate to a full cycle
    original_phi = np.array(phi)
    original_y = np.array(y)
    phi = np.copy(original_phi)
    y = np.copy(original_y)
    for i in range(1, k):
        y = np.concatenate((y, original_y), axis=None)
        original_phi += target_final_phi  # add to every element
        phi = np.concatenate((phi, original_phi), axis=None)

    # necessary transform for normalization
    phi = (-phi - pi) % (2 * pi)  # negate rotation direction and have pi initial phase
    return list(y), center_distance, list(phi)


def cumulative_sum(x: list) -> list:
    length = len(x)
    result = [x[0]]
    for i in range(1, length):
        result.append(result[i - 1] + x[i])
    return result


def to_polygon(sample_function, theta_range=(0, 2 * pi)) -> Polygon:
    range_start, range_end = theta_range
    return Polygon([(r * cos(theta), - r * sin(theta)) for r, theta in
                    zip(sample_function, np.linspace(range_start, range_end, len(sample_function), endpoint=False))])


def rotate_and_cut(drive_polygon: Polygon, center_distance, phi, k=1, debugger: MyDebugger = None,
                   replay_animation: bool = False):
    from shapely.affinity import translate, rotate
    driven_polygon = to_polygon([center_distance] * len(phi))
    delta_theta = 2 * pi / len(phi) * k
    driven_polygon = translate(driven_polygon, center_distance)
    complete_phi = phi + [phi[0]]  # so that it rotates back
    phi_incremental = [0.0] + [complete_phi[i] - complete_phi[i - 1] for i in range(1, len(complete_phi))]
    assert isclose(sum(phi_incremental) % (2 * pi), 0, rel_tol=1e-5)
    angle_sum = 0

    fig, subplot = plt.subplots()
    subplot.set_title('Dual Shape(Cut)')
    subplot.axis('equal')

    plt.ion()
    for index, angle in enumerate(phi_incremental):
        angle_sum = delta_theta * index
        _drive_polygon = rotate(drive_polygon, angle_sum, use_radians=True, origin=(0, 0))
        driven_polygon = rotate(driven_polygon, angle, use_radians=True, origin=(center_distance, 0))
        driven_polygon = driven_polygon.difference(_drive_polygon)
        _plot_polygon((_drive_polygon, driven_polygon))
        plt.scatter((0, center_distance), (0, 0), s=100, c='b')
        if debugger is not None:
            fig.savefig(os.path.join(debugger.get_cutting_debug_dir_name(), f'cutting_{index}.png'))
        plt.pause(0.001)
    assert isclose(angle_sum, 2 * pi * k, rel_tol=1e-5)
    plt.ioff()

    driven_polygon = rotate(driven_polygon, -complete_phi[-1], use_radians=True,
                            origin=(center_distance, 0))  # de-rotate to match phi

    if replay_animation:
        # replay the animation
        plt.ion()
        for index, angle in enumerate(phi):
            theta = delta_theta * index
            _drive_polygon = rotate(drive_polygon, theta, (0, 0), True)
            _driven_polygon = rotate(driven_polygon, angle, (center_distance, 0), True)
            _plot_polygon((_drive_polygon, _driven_polygon))
            plt.scatter((0, center_distance), (0, 0), s=100, c='b')
            plt.pause(0.001)
        plt.ioff()

    driven_polygon = translate(driven_polygon, -center_distance)
    return driven_polygon, fig, subplot


def _plot_polygon(polygons):
    plt.clf()
    for poly in polygons:
        _draw_single_polygon(poly)
    plt.axis('tight')
    plt.axis('equal')
    plt.draw()


def _draw_single_polygon(polygon):
    if not isinstance(polygon, MultiPolygon):
        polygon = polygon,
    for poly in polygon:
        xs, ys = poly.exterior.xy
        plt.plot(xs, ys)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from drive_gears.focal_ellipse_gear import generate_gear
    from shapely.affinity import translate, rotate
    from core.plot_sampled_function import plot_sampled_function

    drive_gear = generate_gear(256)
    y, center_distance, phi = compute_dual_gear(drive_gear, 1)
    plot_sampled_function((drive_gear, y), (phi,), None, 200, 0.001, ((0, 0), (center_distance, 0)), (8, 8),
                          ((-5, 15), (-10, 10)))
    poly, *_ = rotate_and_cut(to_polygon(drive_gear), center_distance, phi, 1, replay_animation=False)
    poly = translate(poly, center_distance)
    poly = rotate(poly, phi[0], origin=(center_distance, 0), use_radians=True)
    _plot_polygon((to_polygon(drive_gear), poly))
    plt.scatter((0, center_distance), (0, 0), s=100, c='b')
    plt.savefig('dual_gear_shapely.png')
    plt.show()
