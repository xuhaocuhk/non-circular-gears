from math import sin, cos, pi, sqrt
from scipy.spatial import distance
import numpy as np


def generate_gear(number_of_samples: int) -> [float]:
    a, b = 1.5, 0.5
    sample_points = np.linspace(0, 2 * pi, number_of_samples)
    return [
        distance.euclidean((0, 0), (a * cos(theta), b * sin(theta)))
        for theta in sample_points
    ]
