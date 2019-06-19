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
from scipy.interpolate import interp1d
import shapely


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

def toEuclideanCoordAsNp(polar_r, center_x, center_y):
    new_x, new_y = toEuclideanCoord(polar_r, center_x, center_y)
    contour = np.concatenate((np.array(new_x).reshape(len(new_x),1), np.array(new_y).reshape(len(new_y),1)), axis=1)
    return contour

# read coutour from a local file
def getSVGShapeAsNp(filename):
    for line in open(filename):
        listWords = line.split(",")
    listWords = np.array(listWords)
    coords = np.array(list(map(lambda word: [float(word.split(" ")[0]), float(word.split(" ")[1])], listWords)))
    return coords

def getIntersDist(p: Point, theta, poly: Polygon, MAX_R):
    outer_point = Point(p.x + MAX_R * sin(theta), p.y + MAX_R * cos(theta))
    ring = LineString(list(poly.exterior.coords))
    inters_pt = ring.intersection(LineString([p, outer_point]))
    return p.distance(inters_pt)

def getMaxIntersDist(p: Point, theta, poly: Polygon, MAX_R):
    outer_point = Point(p.x + MAX_R * sin(theta), p.y + MAX_R * cos(theta))
    ring = LineString(list(poly.exterior.coords))
    inters_pt = ring.intersection(LineString([p, outer_point]))
    if isinstance(inters_pt, shapely.geometry.multipoint.MultiPoint):
        print("XXXSDAFASDFSADF")
        return max([p.distance(ip) for ip in inters_pt])
    else:
        return p.distance(inters_pt)

# get uniform sampled points along boundary
def getUniformCoordinateFunction(contour: np.array):
    polygon = Polygon(contour)
    coord = np.array( list(polygon.exterior.coords) )
    distance = [ np.linalg.norm(coord[i]-coord[i-1]) for i in range(len(coord))]
    cumsum_dist = np.cumsum(distance)
    cumsum_dist = cumsum_dist/cumsum_dist[-1]
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
    vtx = list(poly.exterior.coords)
    distances = [p.distance(Point(v[0], v[1])) for v in vtx]
    MAX_R = max(distances) + 10
    sample_distances = [getMaxIntersDist(p, i * 2 * math.pi / n, poly, MAX_R) for i in range(n)]
    return sample_distances


# read coutour from a local file
def getSVGShape(filename):
    for line in open(filename):
        listWords = line.split(",")
    x_coords = list(map(lambda word: float(word.split(" ")[0]), listWords))
    y_coords = list(map(lambda word: float(word.split(" ")[1]), listWords))
    return x_coords, y_coords



def testSampleVisibleCenters():
    x, y = getSVGShape(filename="../silhouette/mahou.txt")

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
    assert 0 <= x <= 1
    if x < 0.2:
        assert 0 <= x < 0.2
        return height * (x / 0.2)
    elif x < 0.5:
        assert 0.2 <= x < 0.5
        return height
    elif x < 0.7:
        assert 0.5 <= x < 0.7
        return height * (0.7 - x) / 0.2
    else:
        return 0.0


# generate teeth in polar coordinate
def getToothFuc(n: int, tooth_num: int, height: float):
    return [toothShape((i % tooth_num) / tooth_num, height) - height for i in range(n)]


def gen_shapes_different_center():
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


def add_tooth():
    x, y = getSVGShape(filename="../silhouette/mahou.txt")
    n = 1024

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
            polygon = Polygon(zip(x, y))
            plt.figure(figsize=(8, 8))
            plt.axis('equal')
            xx = []
            yy = []
            for i in range(n):
                p = getUniformCoordinateFunction(i / n, polygon)
                #plt.scatter(p[0], p[1], s=50, c='b')
                xx.append(p[0])
                yy.append(p[1])

            new_x = xx
            new_y = yy
            #plt.fill(new_x, new_y, "r", alpha=0.3)


            tooth_func = getToothFuc(n, tooth_num=20, height=20)
            normals = [(new_y[i]-new_y[i+1] , new_x[i+1]-new_x[i]) for i in range(n-1)] # compute normals perpendicular to countour
            normals.append((new_y[n-1] - new_y[0], new_x[0] - new_x[n-1]))
            normals = [ (normals[i][0] / math.sqrt( normals[i][0]*normals[i][0] + normals[i][1]*normals[i][1]) , normals[i][1] / math.sqrt( normals[i][0]*normals[i][0] + normals[i][1]*normals[i][1]))
                        for i in range(n)] # normalization
            deviation = [(normals[i][0]*tooth_func[i], normals[i][1]*tooth_func[i]) for i in range(n)]

            #plt.fill(new_x, new_y, "b", alpha=0.3)
            new_x = [new_x[i] + deviation[i][0] for i in range(n)]
            new_y = [new_y[i] + deviation[i][1] for i in range(n)]

            plt.fill(new_x, new_y, "r", alpha=0.3)
            plt.show()
            input("Stop")
        # else:
        # plt.scatter(x_i, y_i, s=50, c='g')
    plt.show()


def testUniformSampleOnContour():
    x, y = getSVGShape(filename="../silhouette/mahou.txt")
    n = 4096

    polygon = Polygon(zip(x, y))
    poly_bound = polygon.bounds

    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    for i in range(1000):
        p = getUniformCoordinateFunction(i / 1000, polygon)
        plt.scatter(p[0], p[1], s=50, c='b')
    plt.show()
    input("xxx")

def getVisiblePoint(contour):
    polygon = Polygon(contour)
    poly_bound = polygon.bounds
    for i in range(1000):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]

        if isAllVisible(Point(x_i, y_i), polygon):
            return np.array([x_i, y_i])
    return None

def getUniformContourSampledShape(contour: np.array, n: int):
    func = getUniformCoordinateFunction(contour)
    return np.array( [[func(i/n)[0], func(i/n)[1]]  for i in range(n)] )

def addToothToContour(contour: np.array, height: int, tooth_num: int):
    n = len(contour)
    tooth_func = getToothFuc(n, tooth_num=n/tooth_num, height=height)
    normals = [(contour[i][1] - contour[i + 1][1], contour[i + 1][0] - contour[i][0]) for i in range(n - 1)]  # compute normals perpendicular to countour
    normals.append((contour[n - 1][1] - contour[0][1], contour[0][0] - contour[n - 1][0]))
    normals = [(normals[i][0] / math.sqrt(normals[i][0] * normals[i][0] + normals[i][1] * normals[i][1]),
                normals[i][1] / math.sqrt(normals[i][0] * normals[i][0] + normals[i][1] * normals[i][1]))
               for i in range(n)]  # normalization
    deviations = np.array( [ [normals[i][0] * tooth_func[i], normals[i][1] * tooth_func[i]] for i in range(n)] )

    return contour + deviations

def getShapeExample():
    n = 128

    plt.figure(figsize=(8, 8))
    plt.axis('equal')

    # read raw polygon from file
    contour = getSVGShapeAsNp(filename="../silhouette/test.txt")
    assert contour.shape == (len(contour), 2)

    # get center visible point
    center = getVisiblePoint(contour)

    # convert to polar coordinate
    # polar_poly = toPolarCoord(Point(center[0], center[1]), contour, n)
    center = (565, 289)
    polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, n)

    # convert to euclidean coordinate to test
    contour = toEuclideanCoordAsNp(polar_poly, center[0], center[1])

    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, n)

    # add tooth
    contour = addToothToContour(contour, height=10, tooth_num=128)


    plt.fill(contour[:,0], contour[:,1], "r", alpha=0.3)
    plt.scatter(center[0], center[1], s=10, c='b')
    for p in contour:
        plt.scatter(p[0], p[1], s=10, c='b')
    #plt.fill(new_x, new_y, "r", alpha=0.3)
    plt.show()
    input()

if __name__ == '__main__':
    getShapeExample()
