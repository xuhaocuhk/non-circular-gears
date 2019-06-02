import matplotlib.pyplot as plt
import numpy as np


def plot_sampled_function(sample_function: [float], range_start: float, range_end: float):
    sample_points = np.linspace(range_start, range_end, len(sample_function))
    plt.plot(sample_points, sample_function)
    plt.show()
    plt.polar(sample_points, sample_function)
    plt.show()


if __name__ == '__main__':
    from compute_dual_gear import compute_dual_gear
    import math
    import random

    drive_gear = [random.uniform(4, 5) for i in range(8192)]
    plot_sampled_function(drive_gear, 0, 2 * math.pi)
    y, L, phi = compute_dual_gear(drive_gear)
    plot_sampled_function(y, 0, 2 * math.pi)
