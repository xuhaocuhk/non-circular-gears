import numpy as np
from typing import Union, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
from debug_util import MyDebugger
from shape_processor import getUniformContourSampledShape, toExteriorPolarCoord, toCartesianCoordAsNp
from objective_function import triangle_area_representation
from shapely.geometry import Polygon, Point
import itertools
from objective_function import shape_difference_rating
from matplotlib.patches import Rectangle
from core.compute_dual_gear import compute_dual_gear


def counterclockwise_orientation(contour: np.ndarray) -> np.ndarray:
    """
    change a contour to counterclockwise direction
    """
    # using the shoelace formula and code adopted from https://stackoverflow.com/questions/14505565/
    # Wikipedia: det[x_i,x_i+1;y_i,y_i+1]
    shoelace = sum(
        [contour[i - 1, 0] * contour[i, 1] - contour[i, 0] * contour[i - 1, 1] for i in range(contour.shape[0])])
    if shoelace < 0:
        return contour[::-1]
    else:
        return contour


def draw_contour(subplot: Axes, contour: np.ndarray, color: str = 'black', title: str = None):
    if title is not None:
        subplot.set_title(title)
    subplot.plot(*contour.transpose(), color=color)
    subplot.axis('equal')


def sample_result(drive_contour: np.ndarray, drive_polygon: Polygon,
                  sample_window: Tuple[float, float, float, float], k: int) \
        -> Union[Tuple[float, float, float, np.ndarray], None]:
    """
    sample the center of the sample window and get the driven gear
    :param drive_contour: the drive gear contour
    :param drive_polygon: the driving polygon
    :param sample_window: the window in which to take sample (minx, maxx, miny, maxy)
    :param k: the drive/driven ratio
    :return: center_x, center_y, center_distance, the driven gear | None if not possible
    """
    min_x, max_x, min_y, max_y = sample_window
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    if not drive_polygon.contains(Point(center_x, center_y)):
        return None
    polar_coordinates = toExteriorPolarCoord(Point(center_x, center_y), drive_contour, drive_contour.shape[0])
    driven, center_distance, phi = compute_dual_gear(polar_coordinates, k)
    driven_contour = toCartesianCoordAsNp(driven, 0, 0)
    return center_x, center_y, center_distance, driven_contour


def uniform_interval(start, end, count):
    separators = np.linspace(start, end, count + 1, endpoint=True)
    return [(separators[i - 1], separators[i]) for i in range(1, count + 1)]


def sample_drive_gear(drive_contour: np.ndarray, target_driven_contour: np.ndarray, k: int,
                      sampling_count: Tuple[int, int], keep_count: int, comparing_accuracy: int, max_sample_depth: int,
                      debugging_path: str, subplots: Union[List[Axes], None]) \
        -> List[Tuple[float, float, float, float, np.ndarray]]:
    """
    run sampling with respect to the sample drive gear
    :param drive_contour: uniformly sampled drive contour
    :param target_driven_contour: target driven contour, not necessarily uniformly sampled
    :param k: drive/driven ratio
    :param sampling_count: count of samples in each dimension
    :param keep_count: the number of drive gears to be returned
    :param comparing_accuracy: resample rate in compare function
    :param max_sample_depth: maximum depth for resampling
    :param debugging_path: the debugging directory
    :param subplots: subplots to draw the results
    :return:[score, center_x, center_y, center_distance, driven_contour]
    """
    drive_polygon = Polygon(drive_contour)
    min_x, min_y, max_x, max_y = drive_polygon.bounds
    windows = [(min_x, max_x, min_y, max_y)]
    result_pool = []
    for iter_time in range(max_sample_depth):
        # split the windows up
        next_windows = []
        for window in windows:
            min_x, max_x, min_y, max_y = window
            new_windows = itertools.product(uniform_interval(min_x, max_x, sampling_count[0]),
                                            uniform_interval(min_y, max_y, sampling_count[1]))
            new_windows = [(x, y, z, w) for (x, y), (z, w) in new_windows]  # repack
            next_windows += new_windows
        windows = next_windows
        # get results
        result_pool = []
        for index, window in enumerate(windows):
            result = sample_result(drive_contour, drive_polygon, window, k)
            if result is None:
                score = 1e8
                result_pool.append((score, None, None, window))
            else:
                *center, center_distance, result = result
                score = shape_difference_rating(target_driven_contour, result, comparing_accuracy)
                result_pool.append((score, center, center_distance, window))
                if subplots is not None:
                    update_polygon_subplots(drive_contour, result, subplots)
                    min_x, max_x, min_y, max_y = window
                    sample_region = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, color='red', fill=False)
                    subplots[0].add_patch(sample_region)
                    subplots[0].scatter(center[0], center[1], 5)
                    subplots[1].text(0, 0, str(score))
                    plt.savefig(os.path.join(debugging_path, f'{iter_time}_{index}.png'))
        result_pool.sort(key=lambda tup: tup[0])
        result_pool = result_pool[:keep_count]
        windows = [result[3] for result in result_pool]
    result_pool = [(score, center_x, center_y, center_distance, driven_contour) for
                   score, (center_x, center_y), center_distance, driven_contour in result_pool]
    return result_pool


def update_polygon_subplots(drive_contour: np.ndarray, driven_contour: np.ndarray, subplots: List[Axes]):
    for subplot in subplots:
        subplot.clear()
    draw_contour(subplots[0], drive_contour, 'orange', 'Drive Contour')
    draw_contour(subplots[1], driven_contour, 'blue', 'Driven Contour')
    for subplot in subplots:
        subplot.axis('equal')


def sampling_optimization(drive_contour: np.ndarray, driven_contour: np.ndarray, k: int, sampling_count: (int, int),
                          keep_count: int, resampling_accuracy: int, comparing_accuracy: int, debugger: MyDebugger,
                          max_sample_depth: int = 5, max_iteration: int = 1, smoothing: Tuple[int, int] = (0, 0),
                          visualization: Union[Dict, None] = None, draw_tar_functions: bool = False) \
        -> Tuple[float, np.ndarray, np.ndarray]:
    """
    perform sampling optimization for drive contour and driven contour
    :param drive_contour: the driving gear's contour
    :param driven_contour: the driven gear's contour
    :param k: drive/driven ratio
    :param sampling_count: the number of samples in each dimension
    :param keep_count: the count of samples kept
    :param resampling_accuracy: count of points in the sampling procedure
    :param comparing_accuracy: count of samples during comparison
    :param debugger: the debugger for storing data
    :param max_sample_depth: maximum depth for the sampling optimization to use
    :param max_iteration: maximum time for drive/driven to swap and iterate
    :param smoothing: smoothing level to be taken by uniform re-sampling
    :param visualization: None for no figure, otherwise for visualization configuration
    :param draw_tar_functions: True for drawing tar functions in debug windows (affect performance)
    :return: final score, drive contour and driven contour
    """
    drive_contour = counterclockwise_orientation(drive_contour)
    driven_contour = counterclockwise_orientation(driven_contour)
    drive_smoothing, driven_smoothing = smoothing
    drive_contour = getUniformContourSampledShape(drive_contour, resampling_accuracy, drive_smoothing > 0)
    driven_contour = getUniformContourSampledShape(driven_contour, resampling_accuracy, driven_smoothing > 0)
    visualize_config = {
        'fig_size': (16, 9),
    }
    subplots = None
    if visualization is not None:
        visualize_config.update(visualization)
        plt.ion()
        fig, subplots = plt.subplots(3, 2)
        fig.set_size_inches(*visualize_config['fig_size'])
        update_polygon_subplots(drive_contour, driven_contour, subplots[0])

    debugging_root_directory = debugger.get_root_debug_dir_name()
    results = []
    for iteration_count in range(max_iteration):
        debug_directory = os.path.join(debugging_root_directory, f'iteration_{iteration_count}')
        os.makedirs(debug_directory, exist_ok=True)
        results = sample_drive_gear(drive_contour, driven_contour, k, sampling_count, keep_count, comparing_accuracy,
                                    max_sample_depth, debug_directory, subplots[1] if subplots is not None else None)
        for index, result in enumerate(results):
            score, *center, center_distance, driven = result
            if subplots is not None:
                update_polygon_subplots(drive_contour, driven, subplots[1])
                subplots[1][0].scatter(center[0], center[1], 3)
                # TODO: try to find where the center of driven gear is
                subplots[1][1].text(0, 0, str(score))
                if draw_tar_functions:
                    tars = [triangle_area_representation(contour, comparing_accuracy)
                            for contour in (drive_contour, driven)]
                    for subplot, tar in subplots[2], tars:
                        tar = tar[:, 0]
                        subplot.plot(range(len(tar)), tar, color='blue')
                plt.savefig(os.path.join(debug_directory, f'final_result_{index}.png'))
        driven_contour = results[0][3]
        os.makedirs(debug_directory, exist_ok=True)
        drive_contour, driven_contour = driven_contour, drive_contour
        drive_smoothing, driven_smoothing = driven_smoothing, drive_smoothing
        drive_contour = getUniformContourSampledShape(drive_contour, resampling_accuracy, drive_smoothing > 0)
    result = results[0]
    score, *center, center_distance, driven = result
    return score, drive_contour, driven


if __name__ == '__main__':
    square_contour = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
    sampling_optimization(square_contour, square_contour, 1, (5, 5), 5, 1024, 64, MyDebugger(['square', 'square']),
                          visualization={}, draw_tar_functions=True)
