from plot.qt_plot import Plotter
from core.objective_function import shape_difference_rating, trivial_distance
from core.optimize_dual_shapes import counterclockwise_orientation
from random import uniform
import numpy as np
from shapely.geometry import Point, Polygon
from debug_util import MyDebugger
from shape_factory import toCartesianCoordAsNp, toExteriorPolarCoord
from core.compute_dual_gear import compute_dual_gear
from plot.plot_sampled_function import rotate

if __name__ == '__main__':
    plotter = Plotter()
    debugger = MyDebugger('wrong')
    target_square = counterclockwise_orientation(np.array([
        (-5, -5),
        (-5, 5),
        (5, 5),
        (5, -5)
    ]))
    lowest_score = 100000
    best_result = None
    for iteration_time in range(1000):
        square_contour = np.array([
            (uniform(0, 10), uniform(0, 10)),
            (uniform(0, 10), uniform(-10, 0)),
            (uniform(-10, 0), uniform(-10, 0)),
            (uniform(-10, 0), uniform(0, 10)),
        ])
        square_contour = counterclockwise_orientation(square_contour)
        center_point = Point(uniform(-5, 5), uniform(-5, 5))
        square_polygon = Polygon(square_contour)

        if not square_polygon.contains(center_point):
            continue

        square_polar = toExteriorPolarCoord(center_point, square_contour, 1024)
        dual_polar, center_dist, phi = compute_dual_gear(square_polar)
        dual_contour = toCartesianCoordAsNp(dual_polar, center_dist, 0)
        score = shape_difference_rating(square_contour, target_square, 64, distance_function=trivial_distance)
        score += shape_difference_rating(dual_contour, target_square, 64, distance_function=trivial_distance)
        if score < lowest_score:
            lowest_score = score
            best_result = square_polar

        # save the result
        plotter.draw_contours(debugger.file_path('result_' + str(iteration_time) + '_%.5f' % (score,) + '.png'), [
            ('math_drive', toCartesianCoordAsNp(square_polar, 0, 0)),
            ('math_driven', np.array(rotate(list(dual_contour), phi[0], (center_dist, 0))))
        ], [(0, 0), (center_dist, 0)])

    dual_polar, center_dist, phi = compute_dual_gear(best_result)
    dual_contour = toCartesianCoordAsNp(dual_polar, center_dist, 0)
    plotter.draw_contours(debugger.file_path('best_result.png'), [
        ('math_drive', toCartesianCoordAsNp(best_result, 0, 0)),
        ('math_driven', np.array(rotate(list(dual_contour), phi[0], (center_dist, 0))))
    ], [(0, 0), (center_dist, 0)])

    # config:
    # figure_translation = (15, 15)
    # figure_scale = 40  # translation done before scaling
