"""
Dual optimization for two functions with phi-averaging
WARNING: assuming k=1
"""

import numpy as np
from util_functions import standard_deviation_distance, align, save_contour
from typing import Iterable, Tuple, List
from core.compute_dual_gear import compute_dual_gear
from core.phi_shape_average import differentiate_function, pre_process, rebuild_polar
from shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
from shapely.geometry import Point, Polygon
from core.optimize_dual_shapes import uniform_interval, update_polygon_subplots
import itertools
from debug_util import DebuggingSuite
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

Window_T = Tuple[float, float, float, float]
Point_T = Tuple[float, float]
Polar_T = List[float]


def phi_distance(polar_drive: Iterable[float], polar_driven: Iterable[float],
                 distance_function=standard_deviation_distance) -> Tuple[float, List[float], List[float], float, float]:
    """
    Get the distance between two phi functions
    :param polar_drive: polar coordinates of the drive contour
    :param polar_driven: polar coordinates of the driven contour
    :param distance_function: the distance function used to calculate the distance
    :return: the distance between phi functions, two d_phi functions and center distances
    """
    y_drive, dist_drive, phi_drive = compute_dual_gear(polar_drive)
    y_driven, dist_driven, phi_driven = compute_dual_gear(polar_driven)
    y_driven, dist_driven, phi_driven = compute_dual_gear(y_driven)
    d_phi_drive = list(differentiate_function(pre_process(phi_drive)))
    d_phi_driven = list(differentiate_function(pre_process(phi_driven)))
    # d_phi_driven = [1.0 / phi for phi in differentiate_function(pre_process(phi_driven))]

    offset = align(d_phi_drive, d_phi_driven, distance_function=distance_function)
    return distance_function(d_phi_drive, d_phi_driven[offset:] + d_phi_driven[:offset]), d_phi_drive, d_phi_driven, \
           dist_drive, dist_driven


def contour_distance(drive_contour: np.ndarray, drive_center: Tuple[float, float], driven_contour: np.ndarray,
                     driven_center: Tuple[float, float], sampling_accuracy: int = 1024) \
        -> Tuple[float, List[float], List[float], float, float]:
    """
    Determine the distance between two contours
    :return: the distance, two d_phi functions and center distances
    """
    drive_polar = toExteriorPolarCoord(Point(*drive_center), drive_contour, sampling_accuracy)
    driven_polar = toExteriorPolarCoord(Point(*driven_center), driven_contour, sampling_accuracy)
    return phi_distance(drive_polar, driven_polar)


def split_window(window: Window_T, x_split_count: int, y_split_count: int) -> List[Window_T]:
    """
    Split a window up to many
    :param window: (min_x, max_x, min_y, max_y)
    :param x_split_count: splitting count in x
    :param y_split_count: splitting count in y
    :return: the sub-windows
    """
    min_x, max_x, min_y, max_y = window
    result = itertools.product(uniform_interval(min_x, max_x, x_split_count),
                               uniform_interval(min_y, max_y, y_split_count))
    # repack and return
    return [(x, y, z, w) for (x, y), (z, w) in result]


def center_of_window(window: Window_T) -> Tuple[float, float]:
    min_x, max_x, min_y, max_y = window
    return (min_x + max_x) / 2, (min_y + max_y) / 2


def align_and_average(array_a: List, array_b: List) -> List:
    assert len(array_a) == len(array_b)
    offset = align(array_a, array_b)
    return [(array_a[index - offset] + array_b[index]) / 2 for index in range(len(array_a))]


def sample_in_windows(drive_contour: np.ndarray, driven_contour: np.ndarray,
                      drive_windows: List[Window_T], driven_windows: List[Window_T], keep_count: int,
                      debugging_suite: DebuggingSuite, center_determine_function=center_of_window,
                      sampling_accuracy=1024) -> List[Tuple[float, Window_T, Window_T, Polar_T]]:
    """
    find the best sample windows
    :param drive_contour: the drive contour
    :param driven_contour: the driven contour
    :param drive_windows: windows to sample of the drive contour
    :param driven_windows: windows to sample of the driven contour
    :param keep_count: count of the windows to be kept
    :param debugging_suite: the debugging suite
    :param center_determine_function: function to determine from window to center
    :param sampling_accuracy: number of samples when converting to polar contour
    :return: list of (score, drive_window, driven_window, reconstructed_drive)
    """
    window_pairs = itertools.product(drive_windows, driven_windows)
    results = []
    path_prefix = debugging_suite.path_prefix  # store in a directory
    if debugging_suite.figure is not None:
        debugging_suite.figure.clear()  # clear the figure
        plt.figure(debugging_suite.figure.number)
        subplots = debugging_suite.figure.subplots(2, 2)
        update_polygon_subplots(drive_contour, driven_contour, subplots[0])
    else:
        subplots = None
    for index, (drive_window, driven_window) in enumerate(window_pairs):
        center_drive = center_determine_function(drive_window)
        center_driven = center_determine_function(driven_window)
        if not (Polygon(drive_contour).contains(Point(*center_drive)) and Polygon(driven_contour).contains(
                Point(*center_driven))):
            # not good windows
            continue
        distance, d_drive, d_driven, dist_drive, dist_driven = \
            contour_distance(drive_contour, center_drive, driven_contour, center_driven, sampling_accuracy)
        reconstructed_drive = rebuild_polar((dist_drive + dist_driven) / 2, align_and_average(d_drive, d_driven))
        results.append((distance, drive_window, driven_window, list(reconstructed_drive)))
        if subplots is not None:
            update_polygon_subplots(drive_contour, driven_contour, subplots[0])  # clear sample regions
            reconstructed_driven, *_ = compute_dual_gear(list(reconstructed_drive))
            reconstructed_drive_contour = toCartesianCoordAsNp(reconstructed_drive, 0, 0)
            reconstructed_driven_contour = toCartesianCoordAsNp(reconstructed_driven, 0, 0)
            update_polygon_subplots(reconstructed_drive_contour, reconstructed_driven_contour, subplots[1])
            min_x, max_x, min_y, max_y = drive_window
            sample_region = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, color='red', fill=False)
            subplots[0][0].add_patch(sample_region)
            min_x, max_x, min_y, max_y = driven_window
            sample_region = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, color='red', fill=False)
            subplots[0][1].add_patch(sample_region)
            subplots[0][0].scatter(*center_drive, 5)
            subplots[0][1].scatter(*center_driven, 5)
            subplots[0][0].text(0, 0, str(center_drive))
            subplots[0][1].text(0, 0, str(center_driven))
            subplots[1][0].scatter(0, 0, 5)
            subplots[1][1].scatter(0, 0, 5)
            subplots[1][0].text(0, 0, 'dist=' + str(distance))
            plt.savefig(path_prefix + f'{index}.png')
            save_contour(path_prefix + f'{index}_drive.dat', reconstructed_drive_contour)
            save_contour(path_prefix + f'{index}_driven.dat', reconstructed_driven_contour)
    results.sort(key=lambda dist, *_: dist)
    return results[:keep_count]


def sampling_optimization(drive_contour: np.ndarray, driven_contour: np.ndarray, sampling_count: int, keep_count: int,
                          sampling_accuracy: int, iteration_count: int, debugging_suite: DebuggingSuite) \
        -> List[Tuple[float, Polar_T]]:
    drive_polygon = Polygon(drive_contour)
    driven_polygon = Polygon(driven_contour)
    min_x, min_y, max_x, max_y = drive_polygon.bounds
    drive_windows = [(min_x, max_x, min_y, max_y)]
    min_x, min_y, max_x, max_y = driven_polygon.bounds
    driven_windows = [(min_x, max_x, min_y, max_y)]
    x_sample, y_sample = sampling_count

    # start iteration
    results = []  # dummy
    for iteration in range(iteration_count):
        path = debugging_suite.debugger.file_path('iteration_' + str(iteration))
        os.makedirs(path, exist_ok=True)
        drive_windows = list(itertools.chain.from_iterable(
            [split_window(drive_window, x_sample, y_sample) for drive_window in drive_windows]))
        driven_windows = list(itertools.chain.from_iterable(
            [split_window(driven_window, x_sample, y_sample) for driven_window in driven_windows]))
        results = sample_in_windows(drive_contour, driven_contour, drive_windows, driven_windows, keep_count,
                                    debugging_suite.sub_suite(os.path.join(path, 'result_')),
                                    sampling_accuracy=sampling_accuracy)
        _, drive_windows, driven_windows, __ = zip(*results)
        if debugging_suite.plotter is not None:
            for index, final_result in enumerate(results):
                score, *_, reconstructed_drive = final_result
                driven, center_distance, phi = compute_dual_gear(reconstructed_drive)
                debugging_suite.plotter.draw_contours(
                    os.path.join(path, f'final_result_{index}_{"%.6f" % (score,)}.png'),
                    [('carve_drive', toCartesianCoordAsNp(reconstructed_drive, 0, 0)),
                     ('carve_driven', toCartesianCoordAsNp(driven, center_distance, 0))],
                    [(0, 0), (center_distance, 0)])
    results = results[:keep_count]
    results.sort(key=lambda dist, *_: dist)
    results = [(score, reconstructed_drive)
               for score, drive_window, driven_window, reconstructed_drive in results]
    return results


if __name__ == '__main__':
    from drive_gears.generate_standard_shapes import gen_ellipse_gear

    test_drive = gen_ellipse_gear(1024)
    dual, *_ = compute_dual_gear(test_drive)
    print(phi_distance(test_drive, dual))
