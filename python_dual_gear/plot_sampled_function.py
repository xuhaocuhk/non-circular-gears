import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from math import sin, cos


def polar_to_rectangular(sample_function: [float], sample_points: [float]) -> [(float, float)]:
    assert len(sample_points) == len(sample_function)
    return [
        (sample_function[i] * cos(sample_points[i]), sample_function[i] * sin(sample_points[i]))
        for i in range(len(sample_points))
    ]


def plot_sampled_function(sample_functions: ([float]), range_start: float, range_end: float):
    if sample_functions == ():
        return

    fig, subplots = plt.subplots(1, len(sample_functions))
    sample_points = np.linspace(range_start, range_end, len(sample_functions[0]))
    for sample_function, subplot in zip(sample_functions, subplots):
        assert len(sample_function) == len(sample_functions[0])
        polygon_points = polar_to_rectangular(sample_function, sample_points)
        # subplot.figure()
        subplot.fill([item[0] for item in polygon_points], [item[1] for item in polygon_points],
                     facecolor='none', edgecolor='purple', linewidth=1)
        subplot.axis('tight')
        subplot.axis('equal')
        subplot.axis('off')
    plt.show()


def rotate(sample_function: [float]) -> [float]:
    pass


if __name__ == '__main__':
    from compute_dual_gear import compute_dual_gear
    import math
    import random

    drive_gear = [random.uniform(4, 5) for i in range(8192)]
    y, L, phi = compute_dual_gear(drive_gear)
    plot_sampled_function((drive_gear, y), 0, 2 * math.pi)
