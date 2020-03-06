import logging
import os
from math import pi, isclose, cos, sin
from typing import Tuple, Optional, Union, List

import numpy as np
from matplotlib import pyplot as plt
from shapely.affinity import translate
from shapely.geometry import Polygon, Point, MultiPolygon

import util_functions
from core.compute_dual_gear import compute_dual_gear
from drive_gears.models import Model
from drive_gears.shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
from plot.plot_sampled_function import plot_sampled_function, rotate
from plot.qt_plot import Plotter
from report import Reporter


def _plot_polygon(polygons, axis):
    plt.clf()
    for poly in polygons:
        _draw_single_polygon(poly)

    plt.axis(axis)
    # plt.axis('equal')

    plt.draw()


def _draw_single_polygon(polygon):
    if not isinstance(polygon, MultiPolygon):
        polygon = polygon,
    for poly in polygon:
        xs, ys = poly.exterior.xy
        plt.plot(xs, ys)


def polygon_to_contour(draw_config: str, polygon: Union[Polygon, MultiPolygon]) -> List[Tuple[str, np.ndarray]]:
    if not isinstance(polygon, MultiPolygon):
        polygon = polygon,
    return [(draw_config, util_functions.shapely_polygon_to_numpy_contour(poly)) for poly in polygon]


def to_polygon(sample_function, theta_range=(0, 2 * pi)) -> Polygon:
    range_start, range_end = theta_range
    return Polygon([(r * cos(theta), - r * sin(theta)) for r, theta in
                    zip(sample_function, np.linspace(range_start, range_end, len(sample_function), endpoint=False))])


def rotate_and_cut(drive_polygon: Polygon, center_distance, phi, k=1, debugger: Reporter = None,
                   replay_animation: bool = False, plot_x_range: Tuple[float, float] = (-1.5, 3),
                   plot_y_range: Tuple[float, float] = (-2.25, 2.25), save_rate: int = 4,
                   plotter: Optional[Plotter] = None):
    # save_rate: save 1 frame per save_rate frames
    from shapely.affinity import translate, rotate
    driven_polygon = to_polygon([center_distance] * len(phi))
    delta_theta = 2 * pi / len(phi) * k
    driven_polygon = translate(driven_polygon, center_distance)
    complete_phi = phi + [phi[0]]  # so that it rotates back
    phi_incremental = [0.0] + [complete_phi[i] - complete_phi[i - 1] for i in range(1, len(complete_phi))]
    assert isclose(sum(phi_incremental) % (2 * pi), 0, rel_tol=1e-5)
    angle_sum = 0

    fig, subplot = plt.subplots(figsize=(7, 7))

    subplot.set_title('Dual Shape(Cut)')
    subplot.axis('equal')

    plt.ion()
    for index, angle in enumerate(phi_incremental):
        angle_sum = delta_theta * index
        _drive_polygon = rotate(drive_polygon, angle_sum, use_radians=True, origin=(0, 0))
        driven_polygon = rotate(driven_polygon, angle, use_radians=True, origin=(center_distance, 0))
        driven_polygon = driven_polygon.difference(_drive_polygon)
        _plot_polygon((_drive_polygon, driven_polygon), plot_x_range + plot_y_range)
        plt.scatter((0, center_distance), (0, 0), s=100, c='b')
        if debugger is not None and index % save_rate == 0:
            file_path = os.path.join(debugger.get_cutting_debug_dir_name(), f'before_cut_{index // save_rate}.png')
            if plotter is None:
                fig.savefig(file_path)
            else:
                plotter.draw_contours(file_path, polygon_to_contour('carve_drive', _drive_polygon) + polygon_to_contour(
                    'carve_driven', driven_polygon), [(center_distance, 0), (0, 0)])
        plt.pause(0.00001)
    assert isclose(angle_sum, 2 * pi * k, rel_tol=1e-5)
    plt.ioff()

    driven_polygon = rotate(driven_polygon, -complete_phi[-1], use_radians=True,
                            origin=(center_distance, 0))  # de-rotate to match phi

    if replay_animation:
        # replay the animation
        plt.ion()
        for index, angle in enumerate(phi):
            theta = delta_theta * index
            _drive_polygon = rotate(drive_polygon, theta, (0, 0), True)
            _driven_polygon = rotate(driven_polygon, angle, (center_distance, 0), True)
            _plot_polygon((_drive_polygon, _driven_polygon), plot_x_range + plot_y_range)
            plt.scatter((0, center_distance), (0, 0), s=100, c='b')
            if debugger is not None and index % save_rate == 0:
                file_path = os.path.join(debugger.get_cutting_debug_dir_name(), f'after_cut_{index // save_rate}.png')
                if plotter is None:
                    fig.savefig(file_path)
                else:
                    plotter.draw_contours(file_path,
                                          polygon_to_contour('carve_drive', _drive_polygon) + polygon_to_contour(
                                              'carve_driven', _driven_polygon), [(center_distance, 0), (0, 0)])
            plt.pause(0.001)
        plt.ioff()

    driven_polygon = translate(driven_polygon, -center_distance)
    return driven_polygon, fig, subplot


def math_cut(drive_model: Model, cart_drive: np.ndarray, reporter: Reporter, plotter: Optional[Plotter],
             animation=False, center_point: Optional[Tuple[float, float]] = None):
    center = center_point or drive_model.center_point
    polar_math_drive = toExteriorPolarCoord(Point(center[0], center[1]), cart_drive, drive_model.sample_num)
    polar_math_driven, center_distance, phi = compute_dual_gear(polar_math_drive, k=drive_model.k)

    if animation:
        plot_sampled_function((polar_math_drive, polar_math_driven), (phi,), reporter.get_math_debug_dir_name(),
                              100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-0.5, 1.5), (-1.1, 1.1)),
                              plotter=plotter)

    # save figures
    plotter.draw_contours(reporter.file_path('math_drive.png'),
                          [('math_drive', toCartesianCoordAsNp(polar_math_drive, 0, 0))], None)
    plotter.draw_contours(reporter.file_path('math_driven.png'),
                          [('math_driven', toCartesianCoordAsNp(polar_math_driven, 0, 0))], None)
    plotter.draw_contours(reporter.file_path('math_results.png'), [
        ('math_drive', toCartesianCoordAsNp(polar_math_drive, 0, 0)),
        ('math_driven', np.array(
            rotate(list(toCartesianCoordAsNp(polar_math_driven, center_distance, 0)), phi[0], (center_distance, 0))))
    ], [(0, 0), (center_distance, 0)])

    logging.info('math rotate complete')
    logging.info(f'Center Distance = {center_distance}')

    return center_distance, phi, polar_math_drive, polar_math_driven


def rotate_and_carve(cart_drive, center, center_distance, debugger, drive_model, phi, plotter, replay_anim=False,
                     save_anim=False, k=1):
    centered_drive = cart_drive - center
    poly_drive_gear = Polygon(centered_drive).buffer(0)
    poly_driven_gear, cut_fig, subplot = rotate_and_cut(poly_drive_gear, center_distance, phi, k=k,
                                                        debugger=debugger if save_anim else None,
                                                        replay_animation=replay_anim, plotter=plotter)
    poly_driven_gear = translate(poly_driven_gear, center_distance).buffer(0).simplify(1e-5)  # as in generate_gear
    if poly_driven_gear.geom_type == 'MultiPolygon':
        poly_driven_gear = max(poly_driven_gear, key=lambda a: a.area)
    cart_driven_gear = np.array(poly_driven_gear.exterior.coords)
    return cart_driven_gear
