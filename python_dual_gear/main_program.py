from debug_util import MyDebugger, DebuggingSuite
from models import our_models, Model, find_model_by_name
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut
from shapely.affinity import translate
import fabrication
import shape_factory
from plot.plot_util import plot_cartesian_shape, plot_polar_shape, init_plot, plot_contour_and_save
import logging
import sys
from plot.plot_sampled_function import plot_sampled_function, rotate
import yaml
from plot.qt_plot import Plotter
import os
from optimization import optimize_pair_from_config
import itertools
import figure_config
from typing import Optional
from core.optimize_dual_shapes import counterclockwise_orientation, clockwise_orientation
from core.dual_optimization import sampling_optimization

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def math_cut(drive_model: Model, cart_drive: np.ndarray, debugger: MyDebugger, plotter: Optional[Plotter],
             animation=False):
    center = drive_model.center_point
    polar_math_drive = toExteriorPolarCoord(Point(center[0], center[1]), cart_drive, drive_model.sample_num)
    polar_math_driven, center_distance, phi = compute_dual_gear(polar_math_drive, k=drive_model.k)

    if animation:
        plot_sampled_function((polar_math_drive, polar_math_driven), (phi,), debugger.get_math_debug_dir_name(),
                              100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-0.5, 1.5), (-1.1, 1.1)),
                              plotter=plotter)

    # save figures
    plotter.draw_contours(debugger.file_path('math_drive.png'),
                          [('math_drive', toCartesianCoordAsNp(polar_math_drive, 0, 0))], None)
    plotter.draw_contours(debugger.file_path('math_driven.png'),
                          [('math_driven', toCartesianCoordAsNp(polar_math_driven, 0, 0))], None)

    logging.info('math rotate complete')
    logging.info(f'Center Distance = {center_distance}')
    return center_distance, phi, polar_math_drive, polar_math_driven


def main(drive_model: Model, driven_model: Model, do_math_cut=True, math_animation=False,
         reply_cut_anim=False, save_cut_anim=True, opt_config='optimization_config.yaml', ):
    # initialize logging system, configuration files, etc.
    debugger, opt_config, plotter = init(drive_model, driven_model, opt_config)

    # get input polygons
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, plotter)

    # math cutting
    if do_math_cut:
        center_distance, phi, polar_math_drive, polar_math_driven = math_cut(drive_model=drive_model,
                                                                             cart_drive=cart_input_drive,
                                                                             debugger=debugger, plotter=plotter,
                                                                             animation=math_animation)

    # optimization
    center, center_distance, cart_drive = optimize_center(cart_input_drive, cart_input_driven, debugger, opt_config,
                                                          plotter)

    # add teeth
    cart_drive = add_teeth(center, center_distance, debugger, cart_drive, drive_model, plotter)

    # rotate and cut
    cart_driven_gear = rotate_and_carve(cart_drive, center, center_distance, debugger, drive_model, phi, plotter,
                                        replay_anim=reply_cut_anim, save_anim=save_cut_anim)

    # save 2D contour
    fabrication.generate_2d_obj(debugger, 'drive_2d.obj', cart_drive)
    fabrication.generate_2d_obj(debugger, 'driven_2d.obj', cart_driven_gear)

    # generate 3D mesh with axle hole
    fabrication.generate_3D_with_axles(6, debugger.file_path('drive_2d.obj'), debugger.file_path('driven_2d.obj'),
                                       (0, 0), (center_distance, 0), debugger, 6)


def rotate_and_carve(cart_drive, center, center_distance, debugger, drive_model, phi, plotter, replay_anim=False,
                     save_anim=False):
    centered_drive = cart_drive - center
    poly_drive_gear = Polygon(centered_drive)
    poly_drive_gear = poly_drive_gear.buffer(0)  # resolve invalid polygon issues
    poly_driven_gear, cut_fig, subplot = rotate_and_cut(poly_drive_gear, center_distance, phi, k=drive_model.k,
                                                        debugger=debugger if save_anim else None,
                                                        replay_animation=replay_anim, plotter=plotter)
    poly_driven_gear = translate(poly_driven_gear, center_distance).buffer(0).simplify(1e-4)  # as in generate_gear
    if poly_driven_gear.geom_type == 'MultiPolygon':
        poly_driven_gear = max(poly_driven_gear, key=lambda a: a.area)
    cart_driven_gear = np.array(poly_driven_gear.exterior.coords)
    return cart_driven_gear


def optimize_center(cart_input_drive, cart_input_driven, debugger, opt_config, plotter):
    debug_suite = DebuggingSuite(debugger, plotter, plt.figure(figsize=(16, 9)))
    results = sampling_optimization(cart_input_drive, cart_input_driven, opt_config['sampling_count'],
                                    opt_config['keep_count'], opt_config['resampling_accuracy'],
                                    opt_config['max_sample_depth'], debug_suite)
    results.sort(key=lambda total_score, *_: total_score)
    best_result = results[0]
    logging.info(f'Best result with score {best_result[0]}')
    score, polar_drive = best_result
    polar_driven, center_distance, phi = compute_dual_gear(polar_drive)
    drive_contour = toCartesianCoordAsNp(polar_drive, 0, 0)
    driven_contour = toCartesianCoordAsNp(polar_driven, center_distance, 0)
    driven_contour = np.array(rotate(driven_contour, phi[0], (center_distance, 0)))
    plotter.draw_contours(debugger.file_path('optimize_result.png'),
                          [('carve_drive', drive_contour), ('carve_driven', driven_contour)],
                          [(0, 0), (center_distance, 0)])
    return (0, 0), center_distance, toCartesianCoordAsNp(polar_drive, 0, 0)


def add_teeth(center, center_distance, debugger, drive, drive_model, plotter):
    drive = counterclockwise_orientation(drive)
    normals = getNormals(drive, None, center)
    drive = addToothToContour(drive, center, center_distance, normals, height=drive_model.tooth_height,
                              tooth_num=drive_model.tooth_num,
                              plt_axis=None, consider_driving_torque=False,
                              consider_driving_continue=False)
    plotter.draw_contours(debugger.file_path('drive_with_teeth.png'), [('input_driven', drive)], None)
    # fabrication.generate_3d_mesh(debugger, 'drive_with_teeth.obj', drive, 1)
    return drive


def get_inputs(debugger, drive_model, driven_model, plotter):
    cart_drive = shape_factory.get_shape_contour(drive_model, uniform=True, plots=None, smooth=drive_model.smooth)
    cart_driven = shape_factory.get_shape_contour(driven_model, uniform=True, plots=None, smooth=driven_model.smooth)
    plotter.draw_contours(debugger.file_path('input_drive.png'), [('input_drive', cart_drive)], None)
    plotter.draw_contours(debugger.file_path('input_driven.png'), [('input_driven', cart_driven)], None)
    logging.debug('original 3D meshes generated')
    return cart_drive, cart_driven


def init(drive_model, driven_model, opt_config):
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
    return debugger, opt_config, plotter


def generate_all_models():
    for model_drive, model_driven in itertools.product(our_models, our_models):
        drive_tooth_contour, final_gear_contour, debugger = generate_gear(model_drive, model_driven, True, True, True,
                                                                          True)

        # generate fabrication files
        fabrication.generate_2d_obj(debugger, 'drive_tooth.obj', drive_tooth_contour)
        fabrication.generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)


if __name__ == '__main__':
    # generate_all_models()

    main(find_model_by_name('ellipse'), find_model_by_name('ellipse'),
         do_math_cut=True, math_animation=False,
         reply_cut_anim=False, save_cut_anim=False, )
