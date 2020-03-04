"""
Functions used for testing
"""
import logging
from typing import List, Tuple, Dict, Any, Union, Iterable

import numpy
import numpy as np
from matplotlib import pyplot as plt

from core.compute_dual_gear import compute_dual_gear
from core.dual_optimization import sampling_optimization
from report import Reporter, SubprocessReporter, ReportingSuite
from core.optimize_dual_shapes import sampling_optimization
import yaml

from drive_gears.shape_processor import toCartesianCoordAsNp
from plot.plot_sampled_function import rotate
from util_functions import save_contour


def optimize_pair_from_config(drive_contour: np.ndarray, driven_contour: np.ndarray, debugger: Reporter,
                              configuration: Union[str, Dict[str, Any]]) \
        -> List[Tuple[float, float, float, float, float, np.ndarray, np.ndarray]]:
    """
    optimize a pair with the optimization config given
    :param drive_contour: the drive contour to be optimized to
    :param driven_contour: the driven contour to be optimized to
    :param debugger: the debugger to provide the path for storing related files
    :param configuration: optimization config, either as yaml filename or a dictionary. See the yaml file for details.
    :return: final total score, score, center_x, center_y, center_distance, drive contour and driven contour
    """
    if isinstance(configuration, str):
        with open(configuration) as yaml_file:
            configuration = yaml.safe_load(yaml_file)
            configuration['sampling_count'] = tuple(configuration['sampling_count'])
    return sampling_optimization(drive_contour, driven_contour, debugger=debugger, visualization={},
                                 draw_tar_functions=True, **configuration)


def optimization_test(names: List[Iterable[str]], optimize_pairs: List[Tuple[np.ndarray, np.ndarray]], config_file: str,
                      parallel: bool = False):
    """
    test optimization for multiple pairs in sub-processes
    :param names: names for each pair for debug use
    :param optimize_pairs: the pairs to be optimized to
    :param config_file: configuration file for optimization
    :param parallel: run subprocesses in parallel
    :return:
    """
    processes = []
    for name, pair in zip(names, optimize_pairs):
        debugger = Reporter(name)
        process = SubprocessReporter(debugger, optimize_pair_from_config, (*pair, debugger, config_file))
        process.start()
        if parallel:
            processes.append(process)
        else:
            process.join()
    for process in processes:
        process.join()


if __name__ == '__main__':
    from drive_gears.shape_factory import get_shape_contour
    from drive_gears.models import our_models

    models = {
        model.name: get_shape_contour(model, True, None, model.smooth) for model in our_models
    }
    test_case_names = [
        ('australia', 'square'),
        ('united_states', 'square'),
        ('france', 'square'),
    ]
    #
    # optimization_test(test_case_names,
    #                   [(models[drive_name], models[dual_name]) for drive_name, dual_name in test_case_names],
    #                   'optimization_config.yaml',
    #                   False)
    optimize_pair_from_config(models['australia'], models['france'], Reporter(['australia', 'france']),
                              'optimization_config.yaml')


def optimize_center(cart_input_drive, cart_input_driven, debugger, opt_config, plotter, k=1):
    debug_suite = ReportingSuite(debugger, plotter, plt.figure(figsize=(16, 9)))
    results = sampling_optimization(cart_input_drive, cart_input_driven, opt_config['sampling_count'],
                                    opt_config['keep_count'], opt_config['resampling_accuracy'],
                                    opt_config['max_sample_depth'], debug_suite, opt_config['torque_weight'], k=k,
                                    mismatch_penalty=opt_config['mismatch_penalty'])
    results.sort(key=lambda total_score, *_: total_score)
    best_result = results[0]
    logging.info(f'Best result with score {best_result[0]}')
    score, polar_drive = best_result
    polar_driven, center_distance, phi = compute_dual_gear(polar_drive, k)
    drive_contour = toCartesianCoordAsNp(polar_drive, 0, 0)
    driven_contour = toCartesianCoordAsNp(polar_driven, center_distance, 0)
    driven_contour = np.array(rotate(driven_contour, phi[0], (center_distance, 0)))
    plotter.draw_contours(debugger.file_path('optimize_result.png'),
                          [('carve_drive', drive_contour), ('carve_driven', driven_contour)],
                          [(0, 0), (center_distance, 0)])
    save_contour(debugger.file_path('optimized_drive.dat'), drive_contour)
    save_contour(debugger.file_path('optimized_driven.dat'), driven_contour)
    return (0, 0), center_distance, toCartesianCoordAsNp(polar_drive, 0, 0), score