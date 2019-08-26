from typing import Iterable, Collection, SupportsFloat, Sized, Callable
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


if __name__ == '__main__':
    array_0 = [1, 2, 3, 4, 5, 6, 7]
    array_1 = [2, 3, 4, 5, 6, 7, 1]
    array_2 = [7, 1, 2, 3, 4, 5, 6]
    print(align(array_0, array_1, 1))
    print(align(array_0, array_2, 1))
