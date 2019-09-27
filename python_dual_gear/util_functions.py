from typing import Union, Collection, SupportsFloat, Callable, Tuple
import math
import numpy as np
import struct
from shapely.geometry import Polygon, Point


def standard_deviation_distance(x: Collection[SupportsFloat], y: Collection[SupportsFloat]) -> float:
    """
    return the standard deviation sqrt(sum((x[i]-y[i])^2)) for two equal-sized arrays
    :param x: the first collection
    :param y: the second collection
    :return: standard deviation
    """
    assert len(x) == len(y)
    return math.sqrt(sum(((float(x_i) - float(y_i)) ** 2 for x_i, y_i in zip(x, y))) / len(x))


def compress(original_array: np.ndarray, new_size: int) -> np.ndarray:
    """
    compress an ndarray
    :param original_array: the original array
    :param new_size: new size (shall be a factor of the original size)
    :return: compressed array
    """
    assert original_array.shape[0] % new_size == 0
    return original_array[::original_array.shape[0] / new_size]


def extend_part(original_array: Union[np.ndarray, Collection[float]], start_index: int, end_index: int,
                new_size: int) -> np.ndarray:
    """
    extend a part of a long periodic function to the new size
    :param original_array: the original periodic array
    :param start_index: the starting of the part to be extended
    :param end_index: the ending (exclusive) of the part to be extended
    :param new_size: the new size to be extended to
    :return: the part, extended to new_size
    """
    assert end_index > start_index
    assert new_size % (end_index - start_index) == 0
    period = 2 * math.pi
    if isinstance(original_array, np.ndarray):
        n = original_array.shape[0]
    else:
        n = len(original_array)
    start_angle = start_index * period / n
    end_angle = end_index * period / n
    return np.interp(
        np.linspace(start_angle, end_angle, new_size, False),
        np.linspace(0, period, n, False),
        original_array,
        period=period
    )


def align(array_a: Collection, array_b: Collection, stride: int = 1,
          distance_function: Callable = standard_deviation_distance, k: int = 1) -> int:
    """
    align two sized iterables to the position with minimum distance
    :param array_a: sized iterable A
    :param array_b: sized iterable B
    :param stride: positive number as the stride of comparison, the larger the faster and less accurate
    :param distance_function: function to calculate
    :param k: multiplicity of the second array when compared to the first array
    :return: offset: align array_a[0] with array_b[offset]
    """
    if k == 1:
        assert len(array_a) == len(array_b)
        return min([(offset, distance_function(array_a, array_b[offset:] + array_b[:offset])) for offset in
                    range(0, len(array_a), stride)], key=lambda tup: tup[1])[0]
    else:
        assert len(array_b) % k == 0
        b_len = len(array_b) / k
        return min([
            (offset, distance_function(array_a, list(extend_part(array_b, offset, offset + b_len, len(array_a)))))
            for offset in range(0, len(array_b), stride)
        ], key=lambda tup: tup[1])[0]


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


def read_contour(filename: str) -> np.ndarray:
    with open(filename, 'rb') as file:
        data = file.read()
        length, = struct.unpack('i', data[:struct.calcsize('i')])
        array = np.empty((length, 2), dtype=float)
        for index, (x, y) in enumerate(struct.iter_unpack('dd', data[struct.calcsize('i'):])):
            array[index] = x, y
        assert index == length - 1
        return array


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


def point_in_contour(contour: np.ndarray, x: float, y: float) -> bool:
    """
    Determines whether the point is in the contour by shapely
    :param contour: the contour to check
    :param x: x-value of the point
    :param y: y-value of the point
    :return: whether the point is in the contour
    """
    return Polygon(contour).contains(Point(x, y))


if __name__ == '__main__':
    test_array = np.array([
        (0, 0),
        (1, 2),
        (3, 4),
        (12.0, 2.23)
    ])
    save_contour('test.dat', test_array)
    print(read_contour('test.dat'))
