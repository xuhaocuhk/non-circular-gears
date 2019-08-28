from debug_util import MyDebugger
from models import our_models, Model, find_model_by_name
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut
from shapely.affinity import translate
import fabrication
import shape_factory
from plot.plot_util import plot_cartesian_shape, plot_polar_shape, init_plot, plot_contour_and_save
import logging
import sys
from plot.plot_sampled_function import plot_sampled_function
import yaml
from plot.qt_plot import Plotter
import os
from optimization import optimize_pair_from_config
import itertools
import figure_config
from typing import Optional
from core.optimize_dual_shapes import counterclockwise_orientation, clockwise_orientation

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def math_rotate(drive_model: Model, cart_drive: np.ndarray, debugger: MyDebugger, plotter: Optional[Plotter],
                animation=False):
    center = drive_model.center_point
    polar_contour = toExteriorPolarCoord(Point(center[0], center[1]), cart_drive, drive_model.sample_num)
    driven_gear, center_distance, phi = compute_dual_gear(polar_contour, k=drive_model.k)

    if animation:
        plot_sampled_function((polar_contour, driven_gear), (phi,), debugger.get_math_debug_dir_name(),
                              100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-0.5, 1.5), (-1.1, 1.1)),
                              plotter=plotter)

    # save figures
    plotter.draw_contours(debugger.file_path('math_rotate/drive.png'),
                          [('math_drive', toCartesianCoordAsNp(polar_contour, 0, 0))], None)
    plotter.draw_contours(debugger.file_path('math_rotate/driven.png'),
                          [('math_driven', toCartesianCoordAsNp(driven_gear, 0, 0))], None)

    logging.info('math rotate complete')
    logging.info(f'Center Distance = {center_distance}')
    return center_distance, phi


def optimize_dual(drive_model: Model, driven_model: Model, do_math_rotate=False, do_cut_rotate=False,
                  opt_config='optimization_config.yaml', math_animation=False):
    # debugger and logging
    debugger = MyDebugger([model.name for model in (drive_model, driven_model)])
    logging_fh = logging.FileHandler(debugger.file_path('logs.log'), 'w')
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
    logging.getLogger('').addHandler(logging_fh)

    # initialize plotter
    plotter = Plotter()

    # parse config
    if isinstance(opt_config, str) and os.path.isfile(opt_config):
        with open(opt_config) as config_file:
            opt_config = yaml.safe_load(config_file)
            opt_config['sampling_count'] = tuple(opt_config['sampling_count'])
    logging.debug('optimization config parse complete, config:' + repr(opt_config))

    # get the input polygon
    cart_drive = shape_factory.get_shape_contour(drive_model, uniform=True, plots=None, smooth=drive_model.smooth)
    cart_driven = shape_factory.get_shape_contour(driven_model, uniform=True, plots=None, smooth=driven_model.smooth)
    plotter.draw_contours(debugger.file_path('input_drive.png'), [('input_drive', cart_drive)], None)
    plotter.draw_contours(debugger.file_path('input_driven.png'), [('input_driven', cart_driven)], None)
    logging.debug('original 3D meshes generated')

    # do math rotate
    if do_math_rotate:
        center_distance, phi = math_rotate(drive_model=drive_model, cart_drive=cart_drive, debugger=debugger, plotter=plotter, animation=math_animation)

    # optimization
    results = optimize_pair_from_config(cart_drive, cart_driven, debugger, opt_config)
    results.sort(key=lambda total_score, *_: total_score)
    best_result = results[0]
    logging.info(f'Best result with score {best_result[0]}')

    total_score, score, *center, center_distance, drive, driven = best_result
    fabrication.generate_3d_mesh(debugger, 'drive_not_cut.obj', drive, 1)
    fabrication.generate_3d_mesh(debugger, 'driven_not_cut.obj', driven, 1)

    # add teeth
    drive = counterclockwise_orientation(drive)
    normals = getNormals(drive, None, center)
    cart_drive = addToothToContour(drive, center, center_distance, normals, height=drive_model.tooth_height,
                                      tooth_num=drive_model.tooth_num,
                                      plt_axis=None, consider_driving_torque=False,
                                      consider_driving_continue=False)
    plotter.draw_contours(debugger.file_path('drive_with_teeth.png'), [('input_driven', cart_drive)], None)
    fabrication.generate_3d_mesh(debugger, 'drive_with_teeth.obj', cart_drive, 1)

    # rotate and cut
    if do_cut_rotate:
        # get the phi function
        drive_polar = toExteriorPolarCoord(Point(center), drive, opt_config['resampling_accuracy'])
        *_, phi = compute_dual_gear(drive_polar, opt_config['k'])

        # initiate cutting
        centered_drive = cart_drive - center
        drive_gear = Polygon(centered_drive)
        drive_gear = drive_gear.buffer(0)  # resolve invalid polygon issues
        driven_gear_cut, cut_fig, subplot = rotate_and_cut(drive_gear, center_distance, phi, k=drive_model.k,
                                                           debugger=debugger, replay_animation=True, plotter=plotter)
        final_polygon = translate(driven_gear_cut, center_distance).buffer(1).simplify(0.2)  # as in generate_gear
        if final_polygon.geom_type == 'MultiPolygon':
            final_polygon = max(final_polygon, key=lambda a: a.area)
        driven_cut = np.array(final_polygon.exterior.coords)
        fabrication.generate_3d_mesh(debugger, 'drive_cut.obj', centered_drive, 1)
        fabrication.generate_3d_mesh(debugger, 'driven_cut.obj', driven_cut, 1)


def generate_gear(drive_model: Model, driven_model: Model, show_math_anim=False, save_math_anim=False,
                  show_cut_anim=False, save_cut_anim=False):
    debugger = MyDebugger(drive_model.name)
    # create a log file in the debug folder
    logging_fh = logging.FileHandler(debugger.file_path('logs.log'), 'w')
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
    logging.getLogger('').addHandler(logging_fh)

    fig, plts = init_plot()
    drive_contour = shape_factory.get_shape_contour(drive_model, True, plts[0], smooth=drive_model.smooth,
                                                    face_color=figure_config.input_shapes['drive_face'],
                                                    edge_color=figure_config.input_shapes['drive_edge'])
    driven_contour = shape_factory.get_shape_contour(driven_model, True, plts[2], smooth=driven_model.smooth,
                                                     face_color=figure_config.input_shapes['driven_face'],
                                                     edge_color=figure_config.input_shapes['driven_edge'])

    # convert to polar coordinate shape
    center = drive_model.center_point
    if center is None:
        center = getVisiblePoint(drive_contour)
        if center is None:
            logging.error("No visible points found, need manually assign one!")
            exit(1)

    polar_drive = toExteriorPolarCoord(Point(center[0], center[1]), drive_contour, drive_model.sample_num)
    plot_polar_shape(plts[1][0], 'Polar shape', polar_drive, drive_model.center_point, drive_model.sample_num)

    # generate and draw the dual shape
    driven_gear, center_distance, phi = compute_dual_gear(polar_drive, k=drive_model.k)
    plot_polar_shape(plts[0][2], 'Dual shape(Math)', driven_gear, (0, 0), drive_model.sample_num)
    logging.info(f'Center Distance = {center_distance}\n')

    if show_math_anim:
        plot_sampled_function((polar_drive, driven_gear), (phi,),
                              debugger.get_math_debug_dir_name() if save_math_anim else None,
                              100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-0.5, 1.5), (-1.1, 1.1)))

    # calculate normals
    plot_cartesian_shape(plts[1][1], "Normals", drive_contour)
    normals = getNormals(drive_contour, plts[1][1], drive_model.center_point)

    # generate teeth
    drive_contour = addToothToContour(drive_contour, center, center_distance, normals, height=drive_model.tooth_height,
                                      tooth_num=drive_model.tooth_num,
                                      plt_axis=plts[1][1], consider_driving_torque=False,
                                      consider_driving_continue=False)
    plot_cartesian_shape(plts[1][2], 'Add Tooth', drive_contour)

    # cut and generate the cutting dual shape
    drive_tooth_contour = []
    for x, y in drive_contour:
        drive_tooth_contour.append((x - center[0], y - center[1]))
    drive_gear = Polygon(drive_tooth_contour)
    drive_gear = drive_gear.buffer(0)  # resolve invalid polygon issues
    driven_gear_cut, cut_fig, subplot = rotate_and_cut(drive_gear, center_distance, phi, k=drive_model.k,
                                                       debugger=debugger if save_cut_anim else None,
                                                       replay_animation=show_cut_anim)
    final_polygon = translate(driven_gear_cut, center_distance)
    final_polygon = final_polygon.buffer(1)
    final_polygon = final_polygon.simplify(0.2)  # for 3d printing
    if final_polygon.geom_type == 'MultiPolygon':
        final_polygon = max(final_polygon, key=lambda a: a.area)
    final_gear_contour = np.array(final_polygon.exterior.coords)

    cut_fig.savefig(debugger.file_path('cut_final.pdf'))
    fig.savefig(debugger.file_path('shapes.pdf'))

    # TODO: consider tolerance when fabricate
    fabrication.generate_2d_obj(debugger, 'drive.obj', toCartesianCoordAsNp(polar_drive, 0, 0))
    fabrication.generate_2d_obj(debugger, 'driven_math.obj', toCartesianCoordAsNp(driven_gear, 0, 0 + center_distance))
    fabrication.generate_3d_mesh(debugger, 'drive_3d.obj', toCartesianCoordAsNp(polar_drive, 0, 0), 1)
    fabrication.generate_3d_mesh(debugger, 'driven.obj', toCartesianCoordAsNp(driven_gear, 0, 0 + center_distance), 1)

    plt.close('all')

    return drive_tooth_contour, final_gear_contour, debugger


def generate_all_models():
    for model_drive, model_driven in itertools.product(our_models, our_models):
        drive_tooth_contour, final_gear_contour, debugger = generate_gear(model_drive, model_driven, True, True, True,
                                                                          True)

        # generate fabrication files
        fabrication.generate_2d_obj(debugger, 'drive_tooth.obj', drive_tooth_contour)
        fabrication.generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)


if __name__ == '__main__':
    # generate_all_models()

    optimize_dual(find_model_by_name('mahou'), find_model_by_name('circular'),
                  do_math_rotate=True, do_cut_rotate=False, math_animation=False)
