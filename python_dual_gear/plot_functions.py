from core.compute_dual_gear import compute_dual_gear
from util_functions import read_contour
import numpy as np
from math import pi
from typing import List
from shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
import matplotlib.pyplot as plt
from shape_factory import get_shape_contour
from main_program import retrieve_model_from_folder
from shapely.geometry import Point
import os
from core.dual_optimization import contour_distance, rebuild_polar, align_and_average
from util_functions import align, extend_part, standard_deviation_distance
from debug_util import MyDebugger
from util_functions import read_contour
from core.phi_shape_average import pre_process, differentiate_function


def draw_function(function: List[float], filename, axis=(0, 2 * pi, 0, 2 * pi), **kwargs):
    x_values = np.linspace(0, 2 * pi, len(function))
    plt.plot(x_values, function, **kwargs)
    plt.axis(axis)
    if filename is not None:
        plt.savefig(filename)
        plt.clf()


def draw_polar(polar: List[float], filename, axis='tight', **kwargs):
    contour = toCartesianCoordAsNp(polar, 0, 0)
    plt.plot(*contour.transpose(), **kwargs)
    plt.axis(axis)
    if filename is not None:
        plt.savefig(filename)
        plt.clf()


if __name__ == '__main__':
    debugger = MyDebugger('plotting')
    os.chdir(debugger.get_root_debug_dir_name())

    center_drive = (0, 0)
    center_driven = (1.9477950195562563, 0)
    figure = plt.figure(figsize=(16, 16))
    k = 5
    sample_rate = 1280
    assert sample_rate % k == 0

    # input_drive = get_shape_contour(retrieve_model_from_folder('human', 'bell'), True, None)
    # input_driven = get_shape_contour(retrieve_model_from_folder('human', 'candy'), True, None)
    drive_path = r'..\2019-10-02_21-26-22_higher_k_test\(plant)leaf_12_k=5_drive.dat'
    driven_path = r'..\2019-10-02_21-26-22_higher_k_test\(plant)leaf_12_k=5_driven.dat'
    input_drive = read_contour(drive_path)
    input_driven = read_contour(driven_path)

    y, l, phi = compute_dual_gear(toExteriorPolarCoord(Point(*center_drive), input_drive, sample_rate), 1)
    draw_polar(y, 'original_driven_k=1.png')
    draw_function(phi, 'original_drive_phi_k=1.png', color='red')
    phi = pre_process(phi)
    draw_function(phi, 'original_drive_phi_preprocessed_k=1.png', axis='tight', color='red')
    y, l, phi = compute_dual_gear(toExteriorPolarCoord(Point(*center_drive), input_drive, sample_rate), 5)
    draw_polar(y, 'original_driven_k=k.png')
    draw_function(phi, 'original_drive_phi_k=k.png', color='red')
    phi = pre_process(phi)
    draw_function(phi, 'original_drive_phi_preprocessed_k=k.png', axis='tight', color='red')
    distance, d_drive, d_driven, dist_drive, dist_driven = \
        contour_distance(input_drive, center_drive, input_driven, center_driven, sample_rate, k=k)
    draw_function(d_drive, 'original_drive_phi_prime.png', color='red')
    draw_function(d_driven, 'original_driven_phi_prime.png', color='blue')

    offset = align(d_drive, d_driven, k=k)
    dist = standard_deviation_distance(d_drive,
                                       list(extend_part(d_driven, offset, offset + len(d_driven) // k, len(d_drive))))
    print(offset, dist)
    extended_driven = list(extend_part(d_driven, offset, offset + int(len(d_driven) / k), len(d_drive)))
    draw_function(extended_driven, 'original_aligned_and_extended_driven.png', color='blue')

    reconstructed_drive = rebuild_polar((dist_drive + dist_driven) / 2, align_and_average(d_drive, d_driven, k=k))
    draw_function(d_drive, None, color='red')
    draw_function(extended_driven, None, color='blue')
    draw_function(align_and_average(d_drive, d_driven, k=k), 'original_aligned.png', color='purple')

    reconstructed_drive = list(reconstructed_drive)
    draw_polar(reconstructed_drive, 'reconstructed_drive.png')
    y, l, phi = compute_dual_gear(reconstructed_drive, k)
    draw_function(reconstructed_drive, 'optimized_drive_polar.png', color='red')
    draw_function(y, 'optimized_driven_polar.png', color='blue')
    draw_polar(y, 'reconstructed_driven.png')
