from core.compute_dual_gear import compute_dual_gear
from util_functions import read_contour
import numpy as np
from math import pi
from typing import List
from shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
import matplotlib.pyplot as plt
from shape_factory import get_shape_contour
from main_program import retrieve_model_from_folder
import os
from core.dual_optimization import contour_distance, rebuild_polar, align_and_average
from util_functions import align, extend_part, standard_deviation_distance
from debug_util import MyDebugger


def draw_function(function: List[float], filename, **kwargs):
    x_values = np.linspace(0, 2 * pi, len(function))
    plt.plot(x_values, function, **kwargs)
    plt.axis([0, 2 * pi, 0, 2 * pi])
    if filename is not None:
        plt.savefig(filename)
        plt.clf()


if __name__ == '__main__':
    debugger = MyDebugger('plotting')
    os.chdir(debugger.get_root_debug_dir_name())

    center_drive = (0.5252525252525252, 0.6245791245791246)
    center_driven = (-2.7755575615628914e-17, -5.204170427930421e-18)
    figure = plt.figure(figsize=(16, 16))

    input_drive = get_shape_contour(retrieve_model_from_folder('human', 'bell'), True, None)
    input_driven = get_shape_contour(retrieve_model_from_folder('human', 'candy'), True, None)

    distance, d_drive, d_driven, dist_drive, dist_driven = \
        contour_distance(input_drive, center_drive, input_driven, center_driven, 1024, 2)
    draw_function(d_drive, 'original_drive_phi_prime.png', color='red')
    draw_function(d_driven, 'original_driven_phi_prime.png', color='blue')

    offset = align(d_drive, d_driven, k=2)
    dist = standard_deviation_distance(d_drive,
                                       list(extend_part(d_driven, offset, offset + len(d_driven) // 2, len(d_drive))))
    print(offset, dist)
    extended_driven = list(extend_part(d_driven, offset, offset + int(len(d_driven) / 2), len(d_drive)))
    draw_function(extended_driven, 'original_aligned_and_extended_driven.png', color='blue')

    reconstructed_drive = rebuild_polar((dist_drive + dist_driven) / 2, align_and_average(d_drive, d_driven, k=2))
    draw_function(d_drive, None, color='red')
    draw_function(extended_driven, None, color='blue')
    draw_function(align_and_average(d_drive, d_driven, k=2), 'original_aligned.png', color='purple')

    reconstructed_drive = list(reconstructed_drive)
    y, l, phi = compute_dual_gear(reconstructed_drive, 2)
    draw_function(reconstructed_drive, 'optimized_drive_polar.png', color='red')
    draw_function(y, 'optimized_driven_polar.png', color='blue')
