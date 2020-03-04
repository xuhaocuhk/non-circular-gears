from report import Reporter
from drive_gears.models import Model, find_model_by_name
from core.rotate_and_carve import math_cut, rotate_and_carve
import fabrication
from drive_gears import shape_factory
import logging
import sys
from gear_tooth import add_teeth
from optimization.optimization import optimize_center
import yaml
from plot.qt_plot import Plotter
import os
from typing import Optional, Iterable, List
import util_functions

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logger = logging.getLogger(__name__)


def get_inputs(debugger, drive_model, driven_model, plotter):
    cart_drive = shape_factory.get_shape_contour(drive_model, uniform=True, plots=None)
    cart_driven = shape_factory.get_shape_contour(driven_model, uniform=True, plots=None)
    plotter.draw_contours(debugger.file_path('input_drive.png'), [('input_drive', cart_drive)], None)
    plotter.draw_contours(debugger.file_path('input_driven.png'), [('input_driven', cart_driven)], None)
    logging.debug('original 3D meshes generated')
    return cart_drive, cart_driven


def init(models: Iterable[Model], opt_config, additional_debugging_names: Optional[List[str]] = None):
    # reporter and logging
    if additional_debugging_names is None:
        additional_debugging_names = []
    reporter = Reporter([model.name for model in models] + additional_debugging_names)
    logging_fh = logging.FileHandler(reporter.file_path('logs.log'), 'w')
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
    return reporter, opt_config, plotter


def main_stage_one(drive_model: Model, driven_model: Model, opt_config='optimization_config.yaml', k=1):
    # initialize logging system, configuration files, etc.
    opt_config = os.path.join(os.path.dirname(__file__), opt_config)
    reporter, opt_config, plotter = init((drive_model, driven_model), opt_config)
    logger.info(f'Optimizing {drive_model.name} with {driven_model.name}')

    # get input polygons
    cart_input_drive, cart_input_driven = get_inputs(reporter, drive_model, driven_model, plotter)

    # optimization
    center, center_distance, cart_drive, score = optimize_center(cart_input_drive, cart_input_driven, reporter,
                                                                 opt_config, plotter, k=k)
    return score


def main_stage_two():
    dir_path = r"E:\OneDrive - The Chinese University of Hong Kong\research_PhD\non-circular-gear\basic_results\finalist\trump_chicken_leg\iteration_2\final_result_0_drive.dat"
    model_name = "trump"

    drive_model = find_model_by_name(model_name)
    drive_model.center_point = (0, 0)
    debugger = Reporter("stage_2_" + model_name)
    plotter = Plotter()

    # read shape
    cart_input_drive = util_functions.read_contour(dir_path)
    cart_input_drive = shape_factory.uniform_and_smooth(cart_input_drive, drive_model)

    # math cutting
    center_distance, phi, polar_math_drive, polar_math_driven = math_cut(drive_model=drive_model,
                                                                         cart_drive=cart_input_drive,
                                                                         debugger=debugger, plotter=plotter,
                                                                         animation=True)

    # add teeth
    cart_drive = add_teeth((0, 0), center_distance, debugger, cart_input_drive, drive_model, plotter)

    # rotate and cut
    cart_driven_gear = rotate_and_carve(cart_drive, (0, 0), center_distance, debugger, drive_model, phi, plotter,
                                        replay_anim=False, save_anim=False)

    # save 2D contour
    fabrication.generate_2d_obj(debugger, 'drive_2d_(0,0).obj', cart_drive)
    fabrication.generate_2d_obj(debugger, f'driven_2d_({center_distance, 0}).obj', cart_driven_gear)

    # generate 3D mesh with axle hole
    fabrication.generate_3D_with_axles(8, debugger.file_path('drive_2d_(0,0).obj'),
                                       debugger.file_path(f'driven_2d_({center_distance, 0}).obj'),
                                       (0, 0), (center_distance, 0), debugger, 6)


