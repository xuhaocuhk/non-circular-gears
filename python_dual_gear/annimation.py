##### This file is for scripting annimations, and for illustration figure generation
from models import find_model_by_name
import main_program

def dual_shape():
    drive_model = find_model_by_name("fish")
    driven_model = find_model_by_name("square")
    opt_config = 'optimization_config.yaml'

    # initialize logging system, configuration files, etc.
    debugger, opt_config, plotter = main_program.init((drive_model, driven_model), opt_config, ["page_1_anim"])

    # get input polygons
    drive_model.smooth = 0
    cart_input_drive, cart_input_driven = main_program.get_inputs(debugger, drive_model, driven_model, plotter)

    # math cutting
    center_distance, phi, polar_math_drive, polar_math_driven = main_program.math_cut(drive_model=drive_model,
                                                                                      cart_drive=cart_input_drive,
                                                                                      debugger=debugger,
                                                                                      plotter=plotter,
                                                                                      animation=True)

if __name__ == '__main__':
    dual_shape()