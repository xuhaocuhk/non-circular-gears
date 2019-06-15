import matplotlib
from math import sin, cos
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from compute_dual_gear import compute_dual_gear
from plot_sampled_function import plot_sampled_function
from shapely.geometry import Polygon
from shapely.geometry import Point


def computeEuclideanCoord_x(r, theta):
    return r * sin(theta)


def computeEuclideanCoord_y(r, theta):
    return r * cos(theta)


def isAllVisible(p: Point, poly: Polygon):
    vtx = list(poly.exterior.coords)
    for p1 in range(len(vtx)):
        line_center = LineString([p, vtx[p1]])
        if line_center.crosses(poly):
            return False
    return True


'''
convert polar to euclidean coordinate
sample usage:
polar_shape = [2, 1, 2, 1, 2, 1, 2, 1]
x, y = toEuclideanCoord(polar_shape)
'''


def toEuclideanCoord(polar_r, center_x, center_y):
    thetas = [theta * 2 * math.pi / len(polar_r) for theta in range(0, len(polar_r))]
    return list(map(lambda n: n + center_x, map(computeEuclideanCoord_x, polar_r, thetas))), list(
        map(lambda n: n + center_y, map(computeEuclideanCoord_y, polar_r, thetas)))


def getIntersDist(p: Point, theta, poly: Polygon, MAX_R):
    outer_point = Point(p.x + MAX_R * sin(theta), p.y + MAX_R * cos(theta))
    ring = LineString(list(poly.exterior.coords))
    inters_pt = ring.intersection(LineString([p, outer_point]))
    return p.distance(inters_pt)


'''
convert euclidean coordinate shape to polar coordinate
'''


def toPolarCoord(p: Point, poly: Polygon, n: int):
    assert isAllVisible(p, poly)
    vtx = list(poly.exterior.coords)
    distances = [p.distance(Point(v[0], v[1])) for v in vtx]
    MAX_R = max(distances) + 10
    sample_distances = [getIntersDist(p, i * 2 * math.pi / n, poly, MAX_R) for i in range(n)]
    return sample_distances


# read coutour from a local file
def getSVGShape(filename):
    for line in open(filename):
        listWords = line.split(",")
    x_coords = list(map(lambda word: float(word.split(" ")[0]), listWords))
    y_coords = list(map(lambda word: float(word.split(" ")[1]), listWords))
    return x_coords, y_coords


def testSampleVisibleCenters():
    x, y = getSVGShape(filename="..\silhouette\mahou.txt")

    polygon = Polygon(zip(x, y))
    poly_bound = polygon.bounds

    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.fill(x, y, "b", alpha=0.3)
    for i in range(1000):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]

        if isAllVisible(Point(x_i, y_i), polygon):
            plt.scatter(x_i, y_i, s=50, c='b')
        else:
            plt.scatter(x_i, y_i, s=50, c='g')

    plt.show()


def testConvertCoordinate():
    x, y = getSVGShape(filename="../silhouette/man.txt")
	n = 4096

    polygon = Polygon(zip(x, y))
    poly_bound = polygon.bounds

    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    # plt.fill(x, y, "b", alpha=0.1)
    for i in range(1000):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]

        if isAllVisible(Point(x_i, y_i), polygon):
            plt.scatter(x_i, y_i, s=50, c='b')
            polar_poly = toPolarCoord(Point(x_i, y_i), polygon, n)
            new_x, new_y = toEuclideanCoord(polar_poly, x_i, y_i)
            plt.fill(new_x, new_y, "r", alpha=0.3)
            plt.show()
            input("Stop")
        # else:
        # plt.scatter(x_i, y_i, s=50, c='g')
    plt.show()



def toothShape(x: float, height: float):
    assert x >= 0 and x <= 1
    if x < 0.2:
        assert x >= 0 and x < 0.2
        return height * (x / 0.2)
    elif x < 0.5:
        assert x >= 0.2 and x < 0.5
        return height
    elif x < 0.7:
        assert x >= 0.5 and x < 0.7
        return height * (0.7 - x) / 0.2
    else:
        return 0.0


# generate teeth in polar coordinate
def getToothFuc(n: int, tooth_num: int, height: float):
    return [toothShape((i % tooth_num) / tooth_num, height) - height for i in range(n)]def different_center():
    x, y = getSVGShape(filename="../silhouette/mahou.txt")

    polygon = Polygon(zip(x, y))
    poly_bound = polygon.bounds

    for i in range(1000):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]

        if isAllVisible(Point(x_i, y_i), polygon):
            # plt.scatter(x_i, y_i, s=50, c='b')
            polar_poly = toPolarCoord(Point(x_i, y_i), polygon, 8192)
            driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)
            plot_sampled_function((polar_poly, driven_gear), (phi,), None, 100, 0.001, [(0, 0), (center_distance, 0)],
                                  (8, 8), ((-800, 1600), (-1200, 1200)))

if __name__ == '__main__':
    # testSampleVisibleCenters()
    # testConvertCoordinate()
    different_center()
