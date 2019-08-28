from typing import Iterable, Collection, SupportsFloat, Sized, Callable, Tuple
import math
import numpy as np
import struct
from shapely.geometry import Polygon


def standard_deviation_distance(x: Iterable[SupportsFloat], y: Iterable[SupportsFloat]) -> float:
    """
    return the standard deviation sqrt(sum((x[i]-y[i])^2)) for two equal-sized arrays
    :param x: the first iterable
    :param y: the second iterable
    :return: standard deviation
    """
    if issubclass(x.__class__, Sized) and issubclass(y.__class__, Sized):
        assert len(x) == len(y)
    return math.sqrt(sum(((float(x_i) - float(y_i)) ** 2 for x_i, y_i in zip(x, y))))


def align(array_a: Collection, array_b: Collection, stride: int = 1,
          distance_function: Callable = standard_deviation_distance) -> int:
    """
    align two sized iterables to the position with minimum distance
    :param array_a: sized iterable A
    :param array_b: sized iterable B
    :param stride: positive number as the stride of comparison, the larger the faster and less accurate
    :param distance_function: function to calculate
    :return: offset: align array_a[0] with array_b[offset]
    """
    assert len(array_a) == len(array_b)
    return min([(offset, distance_function(array_a, array_b[offset:] + array_b[:offset])) for offset in
                range(0, len(array_a), stride)], key=lambda tup: tup[1])[0]


def pack_contour(contour: np.ndarray) -> bytes:
    assert contour.shape[1] == 2
    result = struct.pack('i', contour.shape[0])
    for point in contour:
        x, y = point
        result += struct.pack('dd', x, y)
    return result


def save_contour(filename: str, contour: np.ndarray):
    with open(filename, 'wb') as file:
        file.write(pack_contour(contour))


def shapely_polygon_to_numpy_contour(polygon: Polygon) -> np.ndarray:
    return np.array(list(zip(*polygon.exterior.xy)))


def eval_function(function: np.ndarray, x_range: Tuple[float, float], x_value: float) -> float:
    """
    evaluate a function at given x-value
    :param function: the values of functions with uniformly sampled x, endpoint not inclusive
    :param x_range: the range of x value for the function
    :param x_value: the x value to evaluate the function at
    :return: the value at that point
    """
    return np.interp(x_value, function, np.linspace(*x_range, len(function), False))


def inverse_function(function: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float],
                     final_value: float) -> np.ndarray:
    """
    calculate the inverse function of a given function
    :param function: the values of functions with uniformly sampled x, endpoint not inclusive
    :param x_range: the range of x value for the function
    :param y_range: the range of y value for the function
    :param final_value: the final value of the function (i.e. y for x=endpoint)
    :return: the inverse function, sampled in y_range and endpoint not inclusive
    """
    n = len(function)
    y_samples = np.linspace(*y_range, n + 1, False)
    return np.interp(y_samples, list(function) + [final_value], np.linspace(*x_range, n + 1, endpoint=True))[:-1]


if __name__ == '__main__':
    x_values = list(range(10))
    y_values = np.array([x * x for x in x_values])
    print(inverse_function(y_values, (0, 10), (0, 100), 100))
