from math import sin, cos, pi, sqrt
from scipy.spatial import distance
import numpy as np


def generate_gear(number_of_samples: int) -> [float]:
    a, b = 1.5, 0.5
    sample_points = np.linspace(0, 2 * pi, number_of_samples, endpoint=False)

    def _radius(theta):
        return a * b / distance.euclidean((0, 0), (b * cos(theta), a * sin(theta)))

    return [
        _radius(theta)
        for theta in sample_points
    ]
