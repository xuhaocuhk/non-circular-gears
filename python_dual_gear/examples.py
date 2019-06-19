# thanks for mahou
import matplotlib.pyplot as plt
from compute_shape_center import *
from compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
import numpy
from math import pi


def pick_center_point(input_file, sample_count, center=None):
    contour = getSVGShapeAsNp(filename=f"../silhouette/{input_file}.txt")
    if center is None:
        for i in range(1000):
            center = getVisiblePoint(contour)
            print(f'center={center}')
            polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, sample_count)
            driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)
            plot_sampled_function((polar_poly, driven_gear), (phi,), None, 100, 0.001, [(0, 0), (center_distance, 0)],
                                  (8, 8), ((-800, 1600), (-1200, 1200)))
    else:
        polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, sample_count)
        driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)
        print(center_distance)
        plot_sampled_function((polar_poly, driven_gear), (phi,), None, 100, 0.001, [(0, 0), (center_distance, 0)],
                              (8, 8), ((-800, 1600), (-1200, 1200)))


def cut_gear(input_file, sample_count, center):
    contour = getSVGShapeAsNp(filename=f"../silhouette/{input_file}.txt")
    print(f'center={center}')

    # get phi
    polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, sample_count)
    driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)

    for i in range(len(phi) - 1):
        plt.scatter(i, (phi[i + 1] - phi[i]) / pi, s=5, c='b')
    plt.show()
    # plt.pause(60)
    print(center_distance)
    # add teeth
    contour = toEuclideanCoordAsNp(polar_poly, center[0], center[1])
    contour = getUniformContourSampledShape(contour, sample_count)
    # contour = addToothToContour(contour, height=20, tooth_num=64)

    # contour -= np.array(center)
    new_contour = []
    for x, y in contour:
        new_contour.append((y - center[1], x - center[0]))
    drive_gear = Polygon(new_contour)
    driven_gear = rotate_and_cut(drive_gear, center_distance, phi)
    translated_driven_gear = translate(driven_gear, center_distance)
    _plot_polygon((drive_gear, translated_driven_gear))
    plt.savefig('tangxiao.png')
    plt.show()


if __name__ == '__main__':
    pick_center_point('test', 4096, (0, 0))
    # input('mahou hao fan')
    cut_gear('test', 8192, (0, 0))
