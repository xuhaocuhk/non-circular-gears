import numpy as np
from math import pi, cos


def generate_gear(number_of_samples: int) -> [float]:
    sample_points = np.linspace(0, 2 * pi, number_of_samples, endpoint=False)
    a, e = 1.5, 0.5

    def _radius(theta):
        return a * (1 - e * e) / (1 + e * cos(theta))

    return [_radius(theta) for theta in sample_points]
