from shape_processor import getUniformContourSampledShape
import numpy as np
from dtw import dtw
import math
import matplotlib.pyplot as plt
from typing import Union


def calculate_area(points):
    points = tuple(points)
    assert len(points) == 3
    matrix = np.append(np.array(points), np.ones((3, 1), np.float64), axis=1)
    return 0.5 * np.linalg.det(matrix)


def triangle_area(points, index, spacing):
    indices = index - spacing, index, index + spacing
    indices = [i % len(points) for i in indices]
    return calculate_area((points[index] for index in indices))


def triangle_area_representation(contour: np.ndarray, sample_count: int) -> np.ndarray:
    """
    calculate the TAR of a given contour
    :param contour: counterclockwise [(x,y)] points
    :param sample_count: number of points to be re-sampled
    :return: TAR(n,ts) as a 2-dim array
    """
    contour = getUniformContourSampledShape(contour, sample_count, False)
    # answer = np.empty((sample_count, (sample_count - 1) // 2))
    # for index in range(sample_count):
    #     for ts in range(1, 1 + answer.shape[1]):
    #         answer[index, ts - 1] = triangle_area(contour, index, ts)
    perimeter = sum([np.linalg.norm(contour[i] - contour[i - 1]) for i in range(len(contour))])
    answer = np.array([[triangle_area(contour, index, ts + 1) for ts in range((sample_count - 1) // 2)]
                       for index in range(sample_count)])
    return answer / perimeter ** 2


def tar_to_distance_matrix(tar_a: np.ndarray, tar_b: np.ndarray) -> np.ndarray:
    assert tar_a.shape == tar_b.shape
    ts = tar_a.shape[1]
    answer = np.empty((tar_a.shape[0], tar_b.shape[0]), dtype=float)
    for i in range(answer.shape[0]):
        for j in range(answer.shape[1]):
            distance_sum = 0
            for k in range(ts):
                distance_sum += abs(tar_a[i, k] - tar_b[j, k])
            distance_sum /= ts
            answer[i, j] = distance_sum
    return answer


def dtw_distance(distance_matrix: np.ndarray, offset: int) -> float:
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
    n = distance_matrix.shape[0]

    def distance(index_a, index_b) -> float:
        index_a = index_a % n
        index_b = (index_b - offset) % n
        return distance_matrix[index_a, index_b]

    return dtw(distance_matrix.shape, distance)[0]


def tar_distance(tar_a: np.ndarray, tar_b: np.ndarray) -> float:
    assert tar_a.shape == tar_b.shape
    distance_matrix = tar_to_distance_matrix(tar_a, tar_b)
    return min([dtw_distance(distance_matrix, offset) for offset in range(tar_a.shape[0])])


def shape_difference_rating(contour_a: np.ndarray, contour_b: np.ndarray,
                            sample_rate: Union[int, None] = None) -> float:
    """
    calculate the shape difference level according to TAR function and DSW
    :param contour_a: the contour A array([(x,y)]) in counterclockwise direction
    :param contour_b: the contour B array([(x,y)]) in counterclockwise direction
    :param sample_rate: the re-sampled rate used in the calculation, None for choosing the maximum of input
    :return: TAR DSW-difference
    """
    if sample_rate is None:
        sample_rate = max(contour_a.shape[0], contour_b.shape[0])
    tar_a = triangle_area_representation(contour_a, sample_rate)
    tar_b = triangle_area_representation(contour_b, sample_rate)
    return tar_distance(tar_a, tar_b)


if __name__ == '__main__':
    contours = [
        np.array([(0, 0), (10, 0), (10, 10), (0, 10)]),
        np.array([(0, 0), (5, 5 * math.sqrt(3)), (5 - 5 * math.sqrt(3), 5 + 5 * math.sqrt(3)), (-5 * math.sqrt(3), 5)]),
        np.array([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
        np.array([(0, 0), (1, 0), (1, 1), (0, 1)]),
        np.array([(5 * math.cos(theta), 5 * math.sin(theta))
                  for theta in np.linspace(0, 2 * math.pi, 1024, endpoint=False)]),
    ]
    count_contours = len(contours)
    fig, subplots = plt.subplots(3, count_contours)
    for contour, subplot_above, subplot_below in zip(contours, subplots[0], subplots[1]):
        x, y = zip(*list(contour), contour[0])
        subplot_above.plot(x, y, color='orange')
        subplot_above.axis('equal')
        sampled = getUniformContourSampledShape(contour, 24, False)
        x, y = zip(*list(sampled), sampled[0])
        subplot_below.plot(x, y, color='red')
        subplot_below.axis('equal')
    tars = [triangle_area_representation(contour, 24) for contour in contours]
    for tar, subplot in zip(tars, subplots[2]):
        tar = tar[:, 0]
        subplot.plot(range(len(tar)), tar, color='blue')
    plt.axis('equal')
    distances = np.array(
        [[shape_difference_rating(contour_a, contour_b, 48) for contour_b in contours] for contour_a in contours])
    print(distances)
