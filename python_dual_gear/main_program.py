from debug_util import MyDebugger, DebuggingSuite
from models import our_models, Model, find_model_by_name, retrieve_models_from_folder, retrieve_model_from_folder
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut
from shapely.affinity import translate
import fabrication
import shape_factory
import logging
import sys
from plot.plot_sampled_function import plot_sampled_function, rotate
import yaml
from plot.qt_plot import Plotter
import os
import itertools
from typing import Optional, Iterable, List, Tuple
from core.optimize_dual_shapes import counterclockwise_orientation
from core.dual_optimization import sampling_optimization, dual_annealing_optimization, split_window, center_of_window, \
    align_and_average, contour_distance, rebuild_polar
from util_functions import save_contour
import traceback
import util_functions
import opt_groups
import matplotlib.pyplot as plt
import time
import datetime
from time import perf_counter_ns

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.ERROR)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger(__name__)


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


def optimize_center(cart_input_drive, cart_input_driven, debugger, opt_config, plotter, k=1):
    debug_suite = DebuggingSuite(debugger, None, None)
    results = sampling_optimization(cart_input_drive, cart_input_driven, opt_config['sampling_count'],
                                    opt_config['keep_count'], opt_config['resampling_accuracy'],
                                    opt_config['max_sample_depth'], debug_suite, opt_config['torque_weight'], k=k,
                                    mismatch_penalty=opt_config['mismatch_penalty'])
    results.sort(key=lambda total_score, *_: total_score)
    best_result = results[0]
    logging.info(f'Best result with score {best_result[0]}')
    score, polar_drive = best_result
    polar_driven, center_distance, phi = compute_dual_gear(polar_drive, k)
    drive_contour = toCartesianCoordAsNp(polar_drive, 0, 0)
    driven_contour = toCartesianCoordAsNp(polar_driven, center_distance, 0)
    driven_contour = np.array(rotate(driven_contour, phi[0], (center_distance, 0)))
    plotter.draw_contours(debugger.file_path('optimize_result.png'),
                          [('carve_drive', drive_contour), ('carve_driven', driven_contour)],
                          [(0, 0), (center_distance, 0)])
    save_contour(debugger.file_path('optimized_drive.dat'), drive_contour)
    save_contour(debugger.file_path('optimized_driven.dat'), driven_contour)
    return (0, 0), center_distance, toCartesianCoordAsNp(polar_drive, 0, 0), score


def optimize_center_annealing(cart_input_drive, cart_input_driven, debugger, opt_config, plotter):
    # compatible with optimize_center
    score, polar_drive = dual_annealing_optimization(cart_input_drive, cart_input_driven)
    polar_driven, center_distance, phi = compute_dual_gear(polar_drive)
    return (0, 0), center_distance, toCartesianCoordAsNp(polar_drive, 0, 0)


def add_teeth(center, center_distance, debugger, drive, drive_model, plotter):
    drive = counterclockwise_orientation(drive)
    normals = getNormals(drive, None, center, normal_filter=True)
    drive = addToothToContour(drive, center, center_distance, normals, height=drive_model.tooth_height,
                              tooth_num=drive_model.tooth_num,
                              plt_axis=None, consider_driving_torque=False,
                              consider_driving_continue=False)
    if plotter is not None:
        plotter.draw_contours(debugger.file_path('drive_with_teeth_before.png'), [('input_driven', drive)], None)

    drive = Polygon(drive).buffer(0).simplify(0.000)
    if drive.geom_type == 'MultiPolygon':
        drive = max(drive, key=lambda a: a.area)
    drive = np.array(drive.exterior.coords)
    if plotter is not None:
        plotter.draw_contours(debugger.file_path('drive_with_teeth.png'), [('input_driven', drive)], None)
    return drive


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
    if additional_debugging_names is None: additional_debugging_names = []
    debugger = MyDebugger([model.name for model in models] + additional_debugging_names)
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


def generate_all_models():
    for model_drive, model_driven in itertools.product(our_models, our_models):
        drive_tooth_contour, final_gear_contour, debugger = generate_gear(model_drive, model_driven, True, True, True,
                                                                          True)

        # generate fabrication files
        fabrication.generate_2d_obj(debugger, 'drive_tooth.obj', drive_tooth_contour)
        fabrication.generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)


def main_stage_one(drive_model: Model, driven_model: Model, do_math_cut=True, math_animation=False,
                   reply_cut_anim=False, save_cut_anim=True, opt_config='optimization_config.yaml', k=1):
    # initialize logging system, configuration files, etc.
    opt_config = os.path.join(os.path.dirname(__file__), opt_config)
    debugger, opt_config, plotter = init((drive_model, driven_model), opt_config)
    logger.info(f'Optimizing {drive_model.name} with {driven_model.name}')
    plt.close('all')
    character_str = f'{drive_model.name}, {driven_model.name}'
    print('starting' + character_str)

    # get input polygons
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, None, uniform=False)
    counts = cart_input_drive.shape[0], cart_input_driven.shape[0]
    start_time = perf_counter_ns()
    cart_input_drive, cart_input_driven = get_inputs(debugger, drive_model, driven_model, None, uniform=True)
    print('pre-processing done for ' + character_str)
    pre_processing = perf_counter_ns()

    # optimization
    center, center_distance, cart_drive, score = optimize_center(cart_input_drive, cart_input_driven, debugger,
                                                                 opt_config, plotter, k=k)
    print('optimization done for ' + character_str)
    optimization = perf_counter_ns()

    rotate_and_cut = 0
    cart_driven_gear = np.array([0.0])
    try:
        cart_drive = add_teeth((0, 0), center_distance, debugger, cart_drive, drive_model, None)

        # rotate and cut
        *_, phi = compute_dual_gear(toExteriorPolarCoord(Point(0, 0), cart_drive, 1024), k)
        cart_driven_gear = rotate_and_carve(cart_drive, (0, 0), center_distance, debugger, drive_model, phi, None,
                                            replay_anim=False, save_anim=False)
        print('rotate_and_carve done for ' + character_str)
        rotate_and_cut = perf_counter_ns()
    except:
        print(f'error in stage two for {drive_model.name}, {driven_model.name}')
        logger.error(f'stage two for {drive_model.name}, {driven_model.name}')

    with open(debugger.file_path('timing_and_statistics.txt'), 'w') as file:
        data = {
            'pre_processing': pre_processing - start_time,
            'optimization': optimization - pre_processing,
            'rotate_and_cut': rotate_and_cut - optimization,
            'counts': counts,
            'follower': cart_driven_gear.shape[0]
        }
        print('\n'.join([f'{key}:{value}' for key, value in data.items()]), file=file)
    return score


def main_stage_two():
    # init
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\square_square\iteration_2\final_result_0_drive.dat"
    # model_name = "square"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\heart_heart\iteration_2\final_result_0_drive.dat"
    # model_name = "heart"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\fish_guo\iteration_2\final_result_0_drive.dat"
    # model_name = "fish"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\drop_heart\iteration_2\final_result_0_drive.dat"
    # model_name = "drop"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\fish_butterfly\iteration_2\final_result_0_drive.dat"
    # model_name = "fish"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\starfish_starfish\iteration_2\final_result_0_drive.dat"
    # model_name = "starfish"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\triangle_qingtianwa\iteration_2\final_result_0_drive.dat"
    # model_name = "triangle"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\trump_chicken_leg\iteration_2\final_result_0_drive.dat"
    # model_name = "trump"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\butterfly_fighter\iteration_2\final_result_0_drive.dat"
    # model_name = "butterfly"
    # dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\boy_girl\iteration_2\final_result_0_drive.dat"
    # model_name = "boy"
    dir_path = r'C:\Projects\gears\python_dual_gear\debug\2019-09-30_19-23-20_bell_(human)candy\optimized_drive.dat'
    model_name = 'bell'
    k = 2

    drive_model = find_model_by_name(model_name)
    drive_model.center_point = (0, 0)
    debugger = MyDebugger("stage_2_" + model_name)
    plotter = Plotter()

    # read shape
    cart_input_drive = util_functions.read_contour(dir_path)
    cart_input_drive = shape_factory.uniform_and_smooth(cart_input_drive, drive_model)

    # math cutting
    # center_distance, phi, polar_math_drive, polar_math_driven = math_cut(drive_model=drive_model,
    #                                                                      cart_drive=cart_input_drive,
    #                                                                      debugger=debugger, plotter=plotter,
    #                                                                      animation=True)

    start_time = perf_counter_ns()
    *_, center_distance, phi = compute_dual_gear(toExteriorPolarCoord(Point(0, 0), cart_input_drive, 1024), k)
    # add teeth
    cart_drive = add_teeth((0, 0), center_distance, debugger, cart_input_drive, drive_model, plotter)

    # rotate and cut
    cart_driven_gear = rotate_and_carve(cart_drive, (0, 0), center_distance, debugger, drive_model, phi, None, k=k,
                                        replay_anim=False, save_anim=False)
    rotate_and_cut = perf_counter_ns()
    print('rotate_and_carve done in' + str(rotate_and_cut - start_time))
    print('count of follower:' + str(cart_driven_gear.shape[0]))

    # save 2D contour
    fabrication.generate_2d_obj(debugger, 'drive_2d_(0,0).obj', cart_drive)
    fabrication.generate_2d_obj(debugger, f'driven_2d_({center_distance, 0}).obj', cart_driven_gear)

    # generate 3D mesh with axle hole
    fabrication.generate_3D_with_axles(8, debugger.file_path('drive_2d_(0,0).obj'),
                                       debugger.file_path(f'driven_2d_({center_distance, 0}).obj'),
                                       (0, 0), (center_distance, 0), debugger, 6)


def optimize_pairs():
    for drive, driven in opt_groups.pairs_to_optimize:
        try:
            main_stage_one(find_model_by_name(drive), find_model_by_name(driven), False, False, True, True)
        except:
            traceback.print_stack()


def optimize_pairs_in_folder(source_folder, dest_folder):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../silhouette/'))
    source_folder = os.path.join(base_dir, source_folder)
    dest_folder = os.path.join(base_dir, dest_folder)
    source_models = retrieve_models_from_folder(source_folder)
    dest_models = retrieve_models_from_folder(dest_folder)

    pairs_to_optimize = list(itertools.product(source_models, dest_models))
    if source_folder == dest_folder:
        existing_names = set(
            tuple(sorted((drive_model.name, driven_model.name))) for drive_model, driven_model in pairs_to_optimize)
        pairs_to_optimize = [(drive_model, driven_model) for drive_model, driven_model in pairs_to_optimize if
                             (driven_model.name, drive_model.name) not in existing_names]

    for source_model, dest_model in pairs_to_optimize:
        try:
            logging.info(f'Playing models drive = {source_model.name}, driven = {dest_model.name}')
            score = main_stage_one(source_model, dest_model, False, False, True, True)
            with open(os.path.abspath(os.path.join(os.path.dirname(__file__), 'debug/scores.log')), 'a') as file:
                print(f'{source_model.name},{dest_model.name},{score}', file=file)
            plt.close('all')
        except Exception:
            print(sys.exc_info())


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
    final_results = [
        # (find_model_by_name('square'), find_model_by_name('square')),
        # (find_model_by_name('ellipse'), find_model_by_name('ellipse')),
        # (find_model_by_name('starfish'), find_model_by_name('starfish')),
        # (find_model_by_name('triangle'), find_model_by_name('qingtianwa')),
        # (find_model_by_name('fish'), find_model_by_name('butterfly')),
        # (find_model_by_name('heart'), find_model_by_name('heart')),
        # (find_model_by_name('hat'), find_model_by_name('trump')),
        # (find_model_by_name('girl'), find_model_by_name('hat')),
        # (find_model_by_name('australia'), find_model_by_name('koala')),
        # (find_model_by_name('boy'), find_model_by_name('girl')),
        # (find_model_by_name('drop'), find_model_by_name('heart')),
        # (find_model_by_name('trump'), find_model_by_name('chicken_leg')),
        (find_model_by_name('bell'), find_model_by_name('human/candy')),
        # (find_model_by_name('dove'), find_model_by_name('dove')),
        # (find_model_by_name('dog'), find_model_by_name('food/bond1')),
        # (find_model_by_name('fishA'), find_model_by_name('animal_sea/fishB')),
        # (find_model_by_name('butterfly'), find_model_by_name('fighter')),
        # (find_model_by_name('pot'), find_model_by_name('shoes'))
    ]
    # for drive, driven in final_results:
    #     try:
    #         main_stage_one(drive, driven, k=1)
    #     except:
    #         logger.error(f'Error for {drive.name}, {driven.name}')
    main_stage_two()
    # main_stage_one(retrieve_model_from_folder('human', 'bell'), retrieve_model_from_folder('human', 'candy'), k=2)
