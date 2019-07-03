from debug_util import MyDebugger
from models import our_models
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
import os
from fabrication import generate_2d_obj
import shape_factory
from plot.plot_util import plot_cartesian_shape, plot_polar_shape, init_plot
import logging
import sys

logging.basicConfig(filename='debug\\info.log', level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    debug_mode = False
    model = our_models[0]
    debugger = MyDebugger(model.name)

    fig, plts = init_plot()
    contour = shape_factory.getShapeContour(model, True, plts)

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
    plot_polar_shape(plts[0][2], 'Dual shape(Math)', driven_gear, (0,0), model.sample_num)
    logging.info(f'Center Distance = {center_distance}\n')
    # plot_sampled_function((polar_poly, driven_gear), (phi,), debugger.get_math_debug_dir_name() if debug_mode else None,
    #                      100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-800, 1600), (-1200, 1200)))

    # calculate normals
    plts[1][1].set_title('Cal Normals')
    plts[1][1].fill(contour[:, 0], contour[:, 1], "g", alpha=0.3)
    normals = getNormals(contour, plts[1][1], model.center_point)
    plts[1][1].axis('equal')

    # generate teeth
    contour = addToothToContour(contour, normals, height=model.tooth_height, tooth_num=model.tooth_num,
                                plt_axis=plts[1][1])
    plts[1][2].set_title('Add Tooth')
    plts[1][2].fill(contour[:, 0], contour[:, 1], "g", alpha=0.3)
    plts[1][2].axis('equal')

    # cut and generate the cutting dual shape
    new_contour = []
    for x, y in contour:
        new_contour.append((x - center[0], y - center[1]))
    drive_gear = Polygon(new_contour)
    drive_gear = drive_gear.buffer(0)  # resolve invalid polygon issues
    driven_gear_cut, cut_fig, subplot = rotate_and_cut(drive_gear, center_distance, phi, k=model.k,
                                                       debugger=debugger if debug_mode else None,
                                                       replay_animation=False)
    final_polygon = translate(driven_gear_cut, center_distance)
    final_polygon = final_polygon.buffer(1)
    final_polygon = final_polygon.simplify(0.2) # for 3d printing
    if final_polygon.geom_type == 'MultiPolygon':
        final_polygon = max(final_polygon, key=lambda a: a.area)
    final_gear_contour = np.array(final_polygon.exterior.coords)

    # generate fabrication files
    generate_2d_obj(debugger, 'drive.obj', toCartesianCoordAsNp(polar_contour, 0, 0))
    generate_2d_obj(debugger, 'drive_tooth.obj', new_contour)
    generate_2d_obj(debugger, 'driven_math.obj', toCartesianCoordAsNp(driven_gear, 0, 0 + center_distance))
    generate_2d_obj(debugger, 'driven_cut.obj', final_gear_contour)

    cut_fig.savefig(debugger.file_path('cut_final.pdf'))
    fig.savefig(debugger.file_path('shapes.pdf'))
