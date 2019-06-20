# thanks for mahou
import matplotlib.pyplot as plt
from compute_shape_center import *
from compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
import numpy
from math import pi


def pick_center_point(input_file, sample_count, center=None):
    contour = getSVGShapeAsNp(filename=f"../silhouette/{input_file}.txt")
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, sample_count)
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


def cut_gear(input_file, sample_count, center, tooth_height, tooth_num):
    contour = getSVGShapeAsNp(filename=f"../silhouette/{input_file}.txt")
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, sample_count)
    print(f'center={center}')

    # get phi
    polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, sample_count)
    driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)

    print(center_distance)
    # add teeth
    contour = addToothToContour(contour, height=tooth_height, tooth_num=tooth_num)

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
    models = [('mahou2', 512, (390, 229), 10, 32),
              ('mahou', 512, (710, 437), 15, 32),
              ('wolf', 512, (300, 300) ,10 , 32),
              ('irregular_circle', 512, (480, 214), 8, 32),
              ('ellipse', 512, (438, 204), 8, 32),
              ('spiral_circle_convex', 512, (470, 206), 8, 32),
              ('man', 4096, (93, 180), 1, 128)]
    chosen = 4
    if models[chosen][2] is None:
        pick_center_point(models[chosen][0], models[chosen][1], models[chosen][2])

    cut_gear(models[chosen][0], models[chosen][1], models[chosen][2] , models[chosen][3], models[chosen][4])

