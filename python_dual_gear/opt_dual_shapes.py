import numpy as np
from typing import Union, Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
from debug_util import MyDebugger
from shape_processor import getUniformContourSampledShape
from objective_function import triangle_area_representation


def counterclockwise_orientation(contour: np.ndarray) -> np.ndarray:
    """
    change a contour to counterclockwise direction
    """
    pass  # TODO
    return contour


def draw_contour(subplot: Axes, contour: np.ndarray, color: str = 'black', title: str = None):
    if title is not None:
        subplot.set_title(title)
    subplot.plot(list(contour.transpose()), color=color)
    subplot.axis('equal')


def sample_drive_gear(drive_contour: np.ndarray, target_driven_contour: np.ndarray, sampling_count: (int, int),
                      keep_count: int, comparing_accuracy: int, max_sample_depth: int, debugging_path: str) \
        -> List[(float, float, float, float, np.ndarray)]:
    """
    run sampling with respect to the sample drive gear
    :param drive_contour: uniformly sampled drive contour
    :param target_driven_contour: target driven contour, not necessarily uniformly sampled
    :param sampling_count: count of samples in each dimension
    :param keep_count: the number of drive gears to be returned
    :param comparing_accuracy: resample rate in compare function
    :param max_sample_depth: maximum depth for resampling
    :return:[score, center_x, center_y, center_distance, driven_contour]
    """
    # TODO: sample in a window and always keep the top keep_count
    pass


def update_polygon_subplots(drive_contour: np.ndarray, driven_contour: np.ndarray, subplots: List[Axes]):
    draw_contour(subplots[0], drive_contour, 'orange', 'Drive Contour')
    draw_contour(subplots[1], driven_contour, 'blue', 'Driven Contour')


def sampling_optimization(drive_contour: np.ndarray, driven_contour: np.ndarray, sampling_count: (int, int),
                          keep_count: int, resampling_accuracy: int, comparing_accuracy: int, debugger: MyDebugger,
                          max_sample_depth: int = 5, max_iteration: int = 1, smoothing: Tuple[int, int] = (0, 0),
                          visualization: Union[Dict, None] = None, draw_tar_functions: bool = False) \
        -> List[(float, np.ndarray, np.ndarray)]:
    """
    perform sampling optimization for drive contour and driven contour
    :param drive_contour: the driving gear's contour
    :param driven_contour: the driven gear's contour
    :param sampling_count: the number of samples in each dimension
    :param keep_count: the count of samples kept
    :param resampling_accuracy: count of points in the sampling procedure
    :param comparing_accuracy: count of samples during comparison
    :param debugger: the debugger for storing data
    :param max_sample_depth: maximum depth for the sampling optimization to use
    :param max_iteration: maximum time for drive/driven to swap and iterate
    :param smoothing: smoothing level to be taken by uniform re-sampling
    :param visualization: None for no figure, otherwise for visualization configuration
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
    for iteration_count in range(max_iteration):
        debug_directory = os.path.join(debugging_root_directory, f'iteration_{iteration_count}')
        results = sample_drive_gear(drive_contour, driven_contour, sampling_count, keep_count, comparing_accuracy,
                                    max_sample_depth, debug_directory)
        for index, result in enumerate(results):
            score, *center, center_distance, driven = result
            update_polygon_subplots(drive_contour, driven, subplots[1])
            subplots[1][0].scatter(center[0], center[1], 3)
            # TODO: try to find where the center of driven gear is
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
