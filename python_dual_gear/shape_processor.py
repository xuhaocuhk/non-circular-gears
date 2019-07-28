from math import sin, cos
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy.interpolate import interp1d
import shapely
from matplotlib.lines import Line2D
import multiprocessing
import gear_tooth

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2  # arbitrary default


def isAllVisible(p: Point, poly: Polygon):
    vtx = list(poly.exterior.coords)
    for p1 in range(len(vtx)):
        line_center = LineString([p, vtx[p1]])
        if line_center.crosses(poly):
            return False
    return True


def computeEuclideanCoord_x(r, theta):
    return r * cos(theta)


def computeEuclideanCoord_y(r, theta):
    return -r * sin(theta)


def toCartesianCoord(polar_r, center_x, center_y):
    thetas = [theta * 2 * math.pi / len(polar_r) for theta in range(0, len(polar_r))]
    return list(map(lambda n: n + center_x, map(computeEuclideanCoord_x, polar_r, thetas))), list(
        map(lambda n: n + center_y, map(computeEuclideanCoord_y, polar_r, thetas)))


def toCartesianCoordAsNp(polar_r, center_x, center_y):
    new_x, new_y = toCartesianCoord(polar_r, center_x, center_y)
    contour = np.concatenate((np.array(new_x).reshape(len(new_x), 1), np.array(new_y).reshape(len(new_y), 1)), axis=1)
    return contour


# read coutour from a local file
def getSVGShapeAsNp(filename):
    for line in open(filename):
        listWords = line.split(",")
    listWords = np.array(listWords)
    coords = np.array(list(map(lambda word: [float(word.split(" ")[0]), float(word.split(" ")[1])], listWords)))
    return coords


def getIntersDist(p: Point, theta, poly: Polygon, MAX_R):
    outer_point = Point(p.x + MAX_R * cos(theta), p.y - MAX_R * sin(theta))
    ring = LineString(list(poly.exterior.coords))
    inters_pt = ring.intersection(LineString([p, outer_point]))
    return p.distance(inters_pt)


def getMaxIntersDist(p: Point, theta, poly: Polygon, MAX_R):
    outer_point = Point(p.x + MAX_R * cos(theta), p.y - MAX_R * sin(theta))
    ring = LineString(list(poly.exterior.coords))
    inters_pt = ring.intersection(LineString([p, outer_point]))
    if isinstance(inters_pt, shapely.geometry.multipoint.MultiPoint):
        return max([p.distance(ip) for ip in inters_pt])
    else:
        return p.distance(inters_pt)


# get uniform sampled points along boundary
def getUniformCoordinateFunction(contour: np.array):
    polygon = Polygon(contour)
    coord = np.array(list(polygon.exterior.coords))
    distance = [np.linalg.norm(coord[i] - coord[i - 1]) for i in range(len(coord))]
    cumsum_dist = np.cumsum(distance)
    cumsum_dist = cumsum_dist / cumsum_dist[-1]
    f = interp1d(cumsum_dist, coord, axis=0)
    return f


'''
convert euclidean coordinate shape to polar coordinate
'''


def toPolarCoord(p: Point, contour: np.array, n: int):
    poly = Polygon(contour)
    assert isAllVisible(p, poly)
    vtx = list(poly.exterior.coords)
    distances = [p.distance(Point(v[0], v[1])) for v in vtx]
    MAX_R = max(distances) + 10
    sample_distances = [getIntersDist(p, i * 2 * math.pi / n, poly, MAX_R) for i in range(n)]
    return sample_distances


'''
convert euclidean coordinate shape to polar coordinate
'''


def toExteriorPolarCoord(p: Point, contour: np.array, n: int):
    poly = Polygon(contour)
    assert poly.contains(p)
    vtx = list(poly.exterior.coords)
    distances = [p.distance(Point(v[0], v[1])) for v in vtx]
    MAX_R = max(distances) + 10
    sample_distances = [getMaxIntersDist(p, i * 2 * math.pi / n, poly, MAX_R) for i in range(n)]
    return sample_distances


def getVisiblePoint(contour):
    polygon = Polygon(contour)
    poly_bound = polygon.bounds
    for i in range(100):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]

        if isAllVisible(Point(x_i, y_i), polygon):
            return np.array([x_i, y_i])
    return None


def getUniformContourSampledShape(contour: np.array, n: int):
    func = getUniformCoordinateFunction(contour)
    return np.array([[func(i / n)[0], func(i / n)[1]] for i in range(n)])


def getNormals(contour: np.array, plt_axis, center, normal_filter=True):
    n = len(contour)
    # compute normals perpendicular to countour
    normals = [(contour[i][1] - contour[i + 1][1], contour[i + 1][0] - contour[i][0]) for i in
               range(n - 1)]
    normals.append((contour[n - 1][1] - contour[0][1], contour[0][0] - contour[n - 1][0]))
    # normalization
    normals = [(normals[i][0] / math.sqrt(normals[i][0] * normals[i][0] + normals[i][1] * normals[i][1]),
                normals[i][1] / math.sqrt(normals[i][0] * normals[i][0] + normals[i][1] * normals[i][1]))
               for i in range(n)]
    # normal filtering
    if normal_filter:
        directions = [(contour[i][0] - center[0], contour[i][1] - center[1]) for i in range(n)]
        normals = [normal if dir[0] * normal[0] + dir[1] * normal[1] > 0 else [normal[0] / 1000, normal[1] / 1000] for
                   dir, normal in zip(directions, normals)]

    # moving average for normal smoothing
    SMOOTH_RANGE = 10
    normal_smoothed = []
    extended_normal = normals[-SMOOTH_RANGE:] + normals + normals[:SMOOTH_RANGE]
    for i in range(n):
        avg_norm = np.mean([math.sqrt(n[0] ** 2 + n[1] ** 2) for n in extended_normal[i: i + 2 * SMOOTH_RANGE]])
        current_norm = math.sqrt(extended_normal[i + SMOOTH_RANGE][0] ** 2 + extended_normal[i + SMOOTH_RANGE][1] ** 2)
        normal_smoothed.append([extended_normal[i + SMOOTH_RANGE][0] / current_norm * avg_norm,
                                extended_normal[i + SMOOTH_RANGE][1] / current_norm * avg_norm])
    normals = normal_smoothed

    # normal visualization
    for i, normal in enumerate(normals):
        start = contour[i]
        normal_l = [normal[0] * 0.05, normal[1] * 0.05]
        end = start + normal_l
        l = Line2D([start[0], end[0]], [start[1], end[1]], linewidth=1)
        plt_axis.add_line(l)

    return normals


def addToothToContour(contour: np.array, center, center_dist, normals, height: int, tooth_num: int, plt_axis,
                      consider_driving_torque=False, consider_driving_continue=False):
    n = len(contour)
    assert n % tooth_num == 0
    samplenum_per_teeth = n / tooth_num

    polar_contour = [np.linalg.norm(p - center) for p in contour]

    tooth_samples = np.full(tooth_num, samplenum_per_teeth, dtype=np.int_)
    tooth_samples = np.cumsum(tooth_samples)
    heights = np.full(tooth_num, height)

    if consider_driving_torque:
        for i in range(10):
            tooth_samples = np.insert(tooth_samples, 0, 0)
            driving_ratios = np.array(
                [gear_tooth.sample_avg(tooth_samples[j], tooth_samples[j + 1], polar_contour, center_dist) for j in
                 range(tooth_num)],
                dtype=np.float_)
            driving_ratios = driving_ratios / np.sum(driving_ratios) * n
            re_indexing = np.cumsum(driving_ratios)
            tooth_samples = np.round(re_indexing).astype('int32')

    if consider_driving_continue:
        tooth_widths = np.diff(np.insert(tooth_samples, 0, 0))
        for j in range(tooth_num):
            zero_front_tooth_samples = np.insert(tooth_samples, 0, 0)
            curr_normal = gear_tooth.normal_mid(zero_front_tooth_samples[j], zero_front_tooth_samples[j + 1], normals)
            curr_center_direction = gear_tooth.point_mid(zero_front_tooth_samples[j], zero_front_tooth_samples[j + 1],
                                                         contour, center)
            sin_theta = np.cross(curr_normal, curr_center_direction) / (
                    np.linalg.norm(curr_normal) * np.linalg.norm(curr_center_direction))
            if sin_theta < 0:
                pass  # heights[j] = height
            else:
                heights[j] = tooth_widths[j] / 100 * (sin_theta / math.sqrt(1 - sin_theta ** 2))

    heights = np.clip(heights, height - 0.01, height + 0.03)

    tooth_func = [gear_tooth.teeth_involute_sin(gear_tooth.get_value_on_tooth_domain(i, tooth_samples),
                                                heights[gear_tooth.get_teeth_idx(i, tooth_samples)], width=0.5) for i in
                  range(n)]

    deviations = np.array([[normals[i][0] * tooth_func[i], normals[i][1] * tooth_func[i]] for i in range(n)])
    return contour + deviations


def getShapeExample():
    n = 4096 * 2

    plt.figure(figsize=(8, 8))
    plt.axis('equal')

    # read raw polygon from file
    contour = getSVGShapeAsNp(filename="../silhouette/spiral_circle.txt")
    # plt.fill(contour[:, 0], contour[:, 1], "b", alpha=0.3)

    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, n)
    # plt.fill(contour[:, 0], contour[:, 1], "r", alpha=0.3)

    # get center visible point
    # center = getVisiblePoint(contour)
    # convert to polar coordinate
    # polar_poly = toPolarCoord(Point(center[0], center[1]), contour, n)
    # center = (565, 289)
    # polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, n)
    # convert to euclidean coordinate to test
    # contour = toEuclideanCoordAsNp(polar_poly, center[0], center[1])

    # add tooth
    contour = addToothToContour(contour, height=5, tooth_num=64)

    plt.fill(contour[:, 0], contour[:, 1], "g", alpha=0.3)
    # plt.scatter(center[0], center[1], s=5, c='b')
    # for p in contour:
    #    plt.scatter(p[0], p[1], s=10, c='b')
    plt.show()
    input()


if __name__ == '__main__':
    getShapeExample()
