from debug_util import MyDebugger
from models import our_models
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
import os
import fabrication
import shape_factory
from plot.plot_util import plot_cartesian_shape, plot_polar_shape, init_plot
import logging
import sys
from plot.plot_sampled_function import plot_sampled_function
from shapely.validation import explain_validity

# writing log to file
logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def generate_gear(model, show_math_anim=False, save_math_anim = False, show_cut_anim = False, save_cut_anim=False):
    debugger = MyDebugger(model.name)

    fig, plts = init_plot()
    contour = shape_factory.getShapeContour(model, True, plts, smooth=model.smooth)

    # convert to polar coordinate shape
    center = model.center_point
    if center is None:
        center = getVisiblePoint(contour)
        if center is None:
            logging.error("No visible point found, need manually assign one!")
            exit(1)

    polar_contour = toExteriorPolarCoord(Point(center[0], center[1]), contour, model.sample_num)
    plot_polar_shape(plts[1][0], 'Polar shape', polar_contour, model.center_point, model.sample_num)

    # generate and draw the dual shape
    driven_gear, center_distance, phi = compute_dual_gear(polar_contour, k=model.k)
    plot_polar_shape(plts[0][2], 'Dual shape(Math)', driven_gear, (0, 0), model.sample_num)
    logging.info(f'Center Distance = {center_distance}\n')

    if show_math_anim:
        plot_sampled_function((polar_contour, driven_gear), (phi,),
                              debugger.get_math_debug_dir_name() if save_math_anim else None,
                              100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-0.5, 1.5), (-1.1, 1.1)))

    # calculate normals
    plot_cartesian_shape(plts[1][1], "Normals", contour)
    normals = getNormals(contour, plts[1][1], model.center_point)

    # generate teeth
    contour = addToothToContour(contour, polar_contour, center_distance, normals, height=model.tooth_height, tooth_num=model.tooth_num,
                                plt_axis=plts[1][1], consider_driving_torque=True, consider_driving_continue=True)
    plot_cartesian_shape(plts[1][2], 'Add Tooth', contour)

    # cut and generate the cutting dual shape
    drive_tooth_contour = []
    for x, y in contour:
        drive_tooth_contour.append((x - center[0], y - center[1]))
    drive_gear = Polygon(drive_tooth_contour)
    drive_gear = drive_gear.buffer(0)  # resolve invalid polygon issues
    driven_gear_cut, cut_fig, subplot = rotate_and_cut(drive_gear, center_distance, phi, k=model.k,
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

    fabrication.generate_2d_obj(debugger, 'drive.obj', toCartesianCoordAsNp(polar_contour, 0, 0))
    fabrication.generate_2d_obj(debugger, 'driven_math.obj', toCartesianCoordAsNp(driven_gear, 0, 0 + center_distance))

    plt.close('all')

    return drive_tooth_contour, final_gear_contour, debugger


def generate_all_models():
    for model in our_models:
        drive_tooth_contour, final_gear_contour, debugger = generate_gear(model, True, True, True, True)

        # generate fabrication files
        fabrication.generate_2d_obj(debugger, 'drive_tooth.obj', drive_tooth_contour)
        fabrication.generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)

if __name__ == '__main__':
    # generate_all_models()

    model = our_models[2]
    drive_tooth_contour, final_gear_contour, debugger = generate_gear(model, show_cut_anim = True, save_cut_anim = True)

    # generate fabrication files
    fabrication.generate_2d_obj(debugger, 'drive_tooth.obj', drive_tooth_contour)
    fabrication.generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)
