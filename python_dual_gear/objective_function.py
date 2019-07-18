from shape_processor import getUniformContourSampledShape
import numpy as np
from dtw import dtw


def calculate_area(points):
    assert len(tuple(points)) == 3
    matrix = np.append(np.array(points), np.ones((3, 1), np.float64), axis=1)
    return 0.5 * np.linalg.det(matrix)


def triangle_area(points, index, spacing):
    indices = index - spacing, index, index + spacing
    indices = [i % len(points) for i in indices]
    return calculate_area((points[index] for index in indices))


def triangle_area_representation(contour: np.ndarray, sample_count: int) -> np.ndarray:
    contour = getUniformContourSampledShape(contour, sample_count)
    answer = np.empty((sample_count, (sample_count - 1) // 2))
    for index in range(sample_count):
        for ts in range(1, 1 + answer.shape[1]):
            answer[index, ts] = triangle_area(contour, index, ts)
    return answer


def tar_to_distance(tar_a: np.ndarray, tar_b: np.ndarray) -> np.ndarray:
    assert tar_a.shape == tar_b.shape
    ts = tar_a.shape[1]
    answer = np.empty(tar_a.shape[0], tar_b.shape[0])
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


if __name__ == '__main__':
    pass
