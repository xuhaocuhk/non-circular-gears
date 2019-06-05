import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from math import sin, cos
from functools import reduce
from matplotlib import patches


def polar_to_rectangular(sample_function: [float], sample_points: [float]) -> [(float, float)]:
    assert len(sample_points) == len(sample_function)
    return [
        (r * cos(theta), r * sin(theta))
        for r, theta in zip(sample_function, sample_points)
    ]


def translation(original_points: [(float, float)], translation_vector: (float, float)) -> [(float, float)]:
    return [(x + translation_vector[0], y + translation_vector[1]) for x, y in original_points]


def rotate(polygon_points: [(float, float)], rotation_angle: float) -> [(float, float)]:
    return [
        (x * cos(rotation_angle) + y * sin(rotation_angle), -x * sin(rotation_angle) + y * cos(rotation_angle))
        for x, y in polygon_points
    ]


def generate_polygon(sample_function, sample_points, rotation_angle=0.0, translation_vector=(0.0, 0.0), **kwargs):
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
    return patches.Polygon(np.array(polygon_points), closed=True, **default_options)


def gear_system(sample_functions, sample_points, rotation_angles=(0.0,), gear_positions=((0.0, 0.0),)):
    assert len(sample_functions) == len(sample_points) and len(sample_points) == len(rotation_angles) \
           and len(rotation_angles) == len(gear_positions)
    patches = []
    for function, points, theta, trans in zip(sample_functions, sample_points, rotation_angles, gear_positions):
        polygon = generate_polygon(function, points, theta, trans)
        patches.append(polygon)
    return patches


def plot_sampled_function(sample_functions: ([float]), range_start: float, range_end: float, rotation_angles=(0.0,),
                          gear_positions=((0.0, 0.0),)):
    if sample_functions == ():
        return
    assert reduce(lambda x, y: len(x) == len(y), sample_functions)
    fig, subplot = plt.subplots()
    sample_points = np.linspace(range_start, range_end, len(sample_functions[0]), endpoint=False)
    sample_points = [sample_points] * len(sample_functions)
    for patch in gear_system(sample_functions, sample_points, rotation_angles, gear_positions):
        subplot.add_patch(patch)
    subplot.axis('tight')
    subplot.axis('equal')
    subplot.axis('off')
    plt.show()


def plot_polygon(subplot, polygon_points):
    polygon = subplot.fill([item[0] for item in polygon_points], [item[1] for item in polygon_points],
                           facecolor='none', edgecolor='purple', linewidth=1)
    return polygon


def initial_animation(subplot, sample_function, sample_points):
    def _initial_animation():
        polygon_points = polar_to_rectangular(sample_function, sample_points)
        return plot_polygon(subplot, polygon_points)

    return _initial_animation


def animation_function(subplot, angle_per_frame, sample_function, sample_points):
    def _animate(frame):
        plt.clf()
        angle = angle_per_frame * frame
        return plot_polygon(subplot, rotate(polar_to_rectangular(sample_function, sample_points), angle))

    return _animate


if __name__ == '__main__':
    from compute_dual_gear import compute_dual_gear
    import math
    from drive_gears.ellipse_gear import generate_gear

    drive_gear = generate_gear(8192)
    driven_gear, center_distance, phi = compute_dual_gear(drive_gear)
    plot_sampled_function((drive_gear,), 0, 2 * math.pi)
    plot_sampled_function((drive_gear, driven_gear), 0, 2 * math.pi, [0.0, 0.0], [(0.0, 0.0), (center_distance, 0.0)])
