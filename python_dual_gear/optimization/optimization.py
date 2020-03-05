"""
Functions used for testing
"""
import logging
from typing import List, Tuple, Dict, Any, Union, Iterable

import numpy as np
from matplotlib import pyplot as plt

from core.compute_dual_gear import compute_dual_gear
from core.dual_optimization import sampling_optimization
from report import Reporter, SubprocessReporter, ReportingSuite
import yaml

from drive_gears.shape_processor import toCartesianCoordAsNp
from plot.plot_sampled_function import rotate
from util_functions import save_contour


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
