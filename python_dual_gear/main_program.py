from gear_tooth import add_teeth
from optimization.optimization import optimize_center
from report import Reporter, ReportingSuite
from drive_gears.models import our_models, Model, find_model_by_name, retrieve_models_from_folder
from drive_gears.shape_processor import *
from core.compute_dual_gear import compute_dual_gear
from core.rotate_and_carve import rotate_and_carve
import fabrication
import drive_gears.shape_factory as shape_factory
import logging
import sys
from plot.plot_sampled_function import rotate
import yaml
from plot.qt_plot import Plotter
import os
from typing import Optional, Iterable, List, Tuple
from core.dual_optimization import align_and_average, contour_distance, rebuild_polar
from util_functions import save_contour
import matplotlib.pyplot as plt
from time import perf_counter_ns

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger(__name__)


def get_inputs(debugger, drive_model, driven_model, plotter, uniform=True):
    cart_drive = shape_factory.get_shape_contour(drive_model, uniform=uniform, plots=None)
    cart_driven = shape_factory.get_shape_contour(driven_model, uniform=uniform, plots=None)
    if plotter is not None:
        plotter.draw_contours(debugger.file_path('input_drive.png'), [('input_drive', cart_drive)], None)
        plotter.draw_contours(debugger.file_path('input_driven.png'), [('input_driven', cart_driven)], None)
    logging.debug('original 3D meshes generated')
    return cart_drive, cart_driven


def init(models: Iterable[Model], opt_config, additional_debugging_names: Optional[List[str]] = None):
    # debugger and logging
    if additional_debugging_names is None:
        additional_debugging_names = []
    debugger = Reporter([model.name for model in models] + additional_debugging_names)
    logging_fh = logging.FileHandler(debugger.file_path('logs.log'), 'w')
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s'))
    logging.getLogger('').addHandler(logging_fh)
    # initialize plotter
    plotter = Plotter()
    # parse config
    if opt_config is not None:
        if isinstance(opt_config, str) and os.path.isfile(opt_config):
            with open(opt_config) as config_file:
                opt_config = yaml.safe_load(config_file)
                opt_config['sampling_count'] = tuple(opt_config['sampling_count'])
    logging.debug('optimization config parse complete, config:' + repr(opt_config))
    return debugger, opt_config, plotter


def main(drive_model: Model, driven_model: Model, do_math_cut=True, math_animation=False,
         reply_cut_anim=False, save_cut_anim=True, opt_config='optimization_config.yaml', k=1):
    # initialize logging system, configuration files, etc.
    opt_config = os.path.join(os.path.dirname(__file__), opt_config)
    debugger, opt_config, plotter = init((drive_model, driven_model), opt_config)
    logger.info(f'Optimizing {drive_model.name} with {driven_model.name}')
    plt.close('all')
    character_str = f'{drive_model.name}, {driven_model.name}'
    print('starting ' + character_str)

    # get input polygons
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, plotter, uniform=False)
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, plotter, uniform=True)
    print('pre-processing done for ' + character_str)

    # optimization
    center, center_distance, cart_drive, score = optimize_center(cart_input_drive, cart_input_driven, debugger,
                                                                 opt_config, plotter, k=k)
    print('optimization done for ' + character_str)
    logger.info(f'score = {score}')

    drive_model.center_point = (0, 0)
    cart_drive = shape_factory.uniform_and_smooth(cart_drive, drive_model)

    *_, center_distance, phi = compute_dual_gear(toExteriorPolarCoord(Point(0, 0), cart_drive, 1024), k)
    # add teeth
    cart_drive = add_teeth((0, 0), center_distance, debugger, cart_drive, drive_model, plotter)

    # rotate and cut
    cart_driven_gear = rotate_and_carve(cart_drive, (0, 0), center_distance, debugger, drive_model, phi, None, k=k,
                                        replay_anim=False, save_anim=False)

    # save 2D contour
    fabrication.generate_2d_obj(debugger, 'drive_2d_(0,0).obj', cart_drive)
    fabrication.generate_2d_obj(debugger, f'driven_2d_({center_distance, 0}).obj', cart_driven_gear)

    # generate 3D mesh with axle hole
    fabrication.generate_3D_with_axles(8, debugger.file_path('drive_2d_(0,0).obj'),
                                       debugger.file_path(f'driven_2d_({center_distance, 0}).obj'),
                                       (0, 0), (center_distance, 0), debugger, 6)


def gradual_average(drive_model: Model, driven_model: Model, drive_center: Tuple[float, float],
                    driven_center: Tuple[float, float], count_of_averages: int):
    """
    Gradually average two contours
    :param drive_model: The drive model
    :param driven_model: The driven model
    :param drive_center: center of drive
    :param driven_center: center of driven
    :param count_of_averages: count of average values
    :return: None
    """
    debugger, opt_config, plotter = init((drive_model, driven_model), 'optimization_config.yaml')
    drive_contour, driven_contour = get_inputs(debugger, drive_model, driven_model, plotter)

    distance, d_drive, d_driven, dist_drive, dist_driven = \
        contour_distance(drive_contour, drive_center, driven_contour, driven_center, 1024)
    for average in np.linspace(0, 1, count_of_averages, True):
        center_dist = dist_drive * 0.5 + dist_driven * 0.5
        reconstructed_drive = rebuild_polar(center_dist, align_and_average(d_drive, d_driven, average))
        reconstructed_driven, center_dist, phi = compute_dual_gear(list(reconstructed_drive))
        reconstructed_drive_contour = toCartesianCoordAsNp(reconstructed_drive, 0, 0)
        reconstructed_driven_contour = toCartesianCoordAsNp(reconstructed_driven, center_dist, 0)
        reconstructed_driven_contour = np.array(rotate(reconstructed_driven_contour, phi[0], (center_dist, 0)))
        average_str = '%1.8f' % average
        plotter.draw_contours(debugger.file_path(average_str + '.png'), [
            ('math_drive', reconstructed_drive_contour),
            ('math_driven', reconstructed_driven_contour)
        ], [(0, 0), (center_dist, 0)])
        save_contour(debugger.file_path(average_str + '_drive.dat'), reconstructed_drive_contour)
        save_contour(debugger.file_path(average_str + '_driven.dat'), reconstructed_driven_contour)


if __name__ == '__main__':
    # the function "fund_model_by_name" automatically find the shape from silhouette directory
    main(find_model_by_name('fish'), find_model_by_name('butterfly'))
