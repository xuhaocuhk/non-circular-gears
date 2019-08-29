import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi
from matplotlib import patches
from plot.qt_plot import Plotter
from typing import Optional
import os


def polar_to_rectangular(sample_function: [float], sample_points: [float]) -> [(float, float)]:
    if sample_points is None:
        sample_points = np.linspace(0, 2 * pi, len(sample_function), endpoint=False)
    assert len(sample_points) == len(sample_function)
    return [
        (r * cos(theta), - r * sin(theta))  # theta is clockwise
        for r, theta in zip(sample_function, sample_points)
    ]


def translation(original_points: [(float, float)], translation_vector: (float, float)) -> [(float, float)]:
    return [(x + translation_vector[0], y + translation_vector[1]) for x, y in original_points]


def rotate(polygon_points: [(float, float)], rotation_angle: float, axis=(0.0, 0.0)) -> [(float, float)]:
    p, q = axis
    return [
        (p + (x - p) * cos(rotation_angle) - (y - q) * sin(rotation_angle),
         q + (x - p) * sin(rotation_angle) + (y - q) * cos(rotation_angle))
        for x, y in polygon_points
    ]


def generate_polygon(sample_function, sample_points, rotation_angle=0.0, translation_vector=(0.0, 0.0), update=None,
                     **kwargs):
    default_options = {
        'edgecolor': 'blue',
        'facecolor': None,
        'fill': False,
        'linewidth': 0.5
    }
    polygon_points = polar_to_rectangular(sample_function, sample_points)
    polygon_points = rotate(polygon_points, rotation_angle)
    polygon_points = translation(polygon_points, translation_vector)
    default_options.update(kwargs)
    if update is None:
        return patches.Polygon(np.array(polygon_points), closed=True, **default_options)
    else:
        update.set_xy(np.array(polygon_points))
        return update


def gear_system(sample_functions, sample_points, rotation_angles=(0.0,), gear_positions=((0.0, 0.0),)):
    assert len(sample_functions) == len(sample_points) and len(sample_points) == len(rotation_angles) \
           and len(rotation_angles) == len(gear_positions)
    patches_collection = []
    for function, points, theta, trans in zip(sample_functions, sample_points, rotation_angles, gear_positions):
        polygon = generate_polygon(function, points, theta, trans)
        patches_collection.append(polygon)
    return patches_collection


def sync_rotation(phi_functions, drive_rotation, base_sample_rate):
    assert len(phi_functions)
    rotation_angles = [drive_rotation]
    for phi in phi_functions:
        assert len(phi) % base_sample_rate == 0
        xp = np.linspace(0, 2 * pi * len(phi) / base_sample_rate, len(phi), endpoint=False)
        angle = np.interp(drive_rotation, xp, phi)
        rotation_angles.append(angle)
    return tuple(rotation_angles)


def plot_frame(subplot, sample_functions: ([float],), range_start: float, range_end: float, phi_functions: [[float]],
               drive_rotation: float, gear_positions=((0.0, 0.0),), save=None, plotter: Optional[Plotter] = None):
    if sample_functions == ():
        return
    assert len(phi_functions) == len(sample_functions) - 1
    sample_points = [np.linspace(range_start, range_end, len(sample_function), endpoint=False) for sample_function in
                     sample_functions]
    gear_contours = []
    for patch in gear_system(sample_functions, sample_points,
                             sync_rotation(phi_functions, drive_rotation, len(sample_functions[0])), gear_positions):
        subplot.add_patch(patch)
        if plotter is not None:
            # remember the patches
            gear_contours.append(patch.xy)
    for gear_position in gear_positions:
        plt.scatter(gear_position[0], gear_position[1], s=5, c='b')
    plt.draw()
    if save is not None:
        # WARNING: saving to png will cause the animation to be not smooth (due to IO and image processing)
        if plotter is None:
            # save with matplotlib
            plt.savefig(f'{save}.png')
        else:
            plotter.draw_contours(f'{save}.png', zip(('math_drive', 'math_driven'), gear_contours), gear_positions)


def plot_sampled_function(sample_functions, phi_functions, filename_prefix, frames, frame_pause, gear_positions,
                          figure_size, canvas_limit, angle_range=(0, 2 * pi), plotter: Optional[Plotter] = None):
    fig, subplot = plt.subplots(figsize=figure_size)
    plt.ion()
    xlim, ylim = canvas_limit
    range_start, range_end = angle_range
    frame_count = 0
    for drive_rotation in np.linspace(range_start, range_end, frames, False):
        subplot.clear()
        subplot.set_xlim(xlim)
        subplot.set_ylim(ylim)
        subplot.axis('off')
        frame_count = frame_count + 1
        if filename_prefix is None:
            plot_frame(subplot, sample_functions, range_start, range_end, phi_functions, drive_rotation, gear_positions,
                       plotter=plotter)
        else:
            plot_frame(subplot, sample_functions, range_start, range_end, phi_functions, drive_rotation, gear_positions,
                       os.path.join(filename_prefix, 'cut-%05d' % (frame_count)), plotter=plotter)
        plt.pause(frame_pause)
    plt.ioff()
    plt.close(fig)


if __name__ == '__main__':
    from core.compute_dual_gear import compute_dual_gear
    from debug_util import MyDebugger
    import shape_factory
    from models import find_model_by_name
    from shape_processor import toExteriorPolarCoord
    from shapely.geometry import Point

    drive_model = find_model_by_name('ellipse')
    drive_contour = shape_factory.get_shape_contour(drive_model, True, None, smooth=drive_model.smooth)
    drive_gear = toExteriorPolarCoord(Point(*drive_model.center_point), drive_contour, 1024)

    debugger = MyDebugger('different_k')
    for k in (10, 2, 3, 4, 5, 1):
        driven_gear, center_distance, phi = compute_dual_gear(drive_gear, k)
        os.makedirs(debugger.file_path(f'k={k}'), exist_ok=True)
        plot_sampled_function((drive_gear, driven_gear), (phi,), debugger.file_path(f'k={k}'), 100, 0.001,
                              [(0, 0), (center_distance, 0)], (8, 8), ((-3, 7), (-5, 5)), plotter=Plotter())
        # figure configuration for this:
        # figure_translation = (1.0, 5.0)
        # figure_scale = 200
