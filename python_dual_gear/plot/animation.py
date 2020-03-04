"""
This file is for scripting animations, and for illustration figure generation
"""
import core.compute_dual_gear
import core.rotate_and_carve
from drive_gears.models import find_model_by_name, Model
import main_program
from drive_gears import shape_factory
from main_program import initialize
import logging
from shapely.geometry import Polygon, Point
from core.dual_optimization import split_window, center_of_window
from util_functions import point_in_contour
import numpy as np
from drive_gears.shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
from core.compute_dual_gear import compute_dual_gear
from plot.plot_sampled_function import rotate


def dual_shape():
    drive_model = find_model_by_name("fish")
    driven_model = find_model_by_name("square")
    opt_config = 'optimization_config.yaml'

    # initialize logging system, configuration files, etc.
    debugger, opt_config, plotter = main_program.initialize((drive_model, driven_model), opt_config, ["page_1_anim"])

    # get input polygons
    drive_model.smooth = 0
    cart_input_drive, cart_input_driven = main_program.get_inputs(debugger, drive_model, driven_model, plotter)

    # math cutting
    center_distance, phi, polar_math_drive, polar_math_driven = core.rotate_and_carve.math_cut(drive_model=drive_model,
                                                                                               cart_drive=cart_input_drive,
                                                                                               reporter=debugger,
                                                                                               plotter=plotter,
                                                                                               animation=True)


def get_duals(drive_model: Model, x_sample_count: int, y_sample_count: int, horizontal_shifting: float):
    """
    Get duals of a given drive model, self-creating debugger
    :param drive_model: the driving model
    :param x_sample_count: count of samples in x direction
    :param y_sample_count: count of samples in y direction
    :param horizontal_shifting: shifting in x direction to keep the drive away from input
    :return: None
    """
    debugger, _, plotter = initialize((drive_model,), None, ['duals'])
    drive_contour = shape_factory.get_shape_contour(drive_model, True, None, drive_model.smooth)
    logging.debug('drive model loaded')

    # get the bounding
    drive_polygon = Polygon(drive_contour)
    min_x, min_y, max_x, max_y = drive_polygon.bounds
    drive_windows = [(min_x, max_x, min_y, max_y)]
    drive_windows = split_window(drive_windows[0], x_sample_count, y_sample_count)
    centers = [center_of_window(window) for window in drive_windows]

    # start finding the dual
    for index, center in enumerate(centers):
        if not point_in_contour(drive_contour, *center):
            logging.info(f'Point #{index}{center} not in contour')
            continue

        drive_polar = toExteriorPolarCoord(Point(*center), drive_contour, 1024)
        driven_polar, center_distance, phi = compute_dual_gear(drive_polar)
        drive_new_contour = toCartesianCoordAsNp(drive_polar, horizontal_shifting, 0)
        driven_contour = toCartesianCoordAsNp(driven_polar, horizontal_shifting + center_distance, 0)
        driven_contour = np.array(rotate(driven_contour, phi[0], (horizontal_shifting + center_distance, 0)))

        # move things back to center
        drive_new_contour += np.array((center[0], center[1]))
        driven_contour += np.array((center[0], center[1]))

        plotter.draw_contours(debugger.file_path(f'{index}.png'), [
            ('input_drive', drive_contour),
            ('math_drive', drive_new_contour),
            ('math_driven', driven_contour)
        ], [(horizontal_shifting + center[0], center[1]),
            (horizontal_shifting + center_distance + center[0], center[1])])


if __name__ == '__main__':
    dual_shape()
