from models import our_models
from examples import cut_gear
from compute_shape_center import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
from debug_util import MyDebugger
from models import Model
from core.plot_sampled_function import plot_sampled_function
import os
import logging
from matplotlib.lines import Line2D

if __name__ == '__main__':
    debug_mode = False

    model = our_models[1]
    debugger = MyDebugger(model.name)

    # set up the plotting window
    fig, plts = plt.subplots(2, 3)
    fig.set_size_inches(16, 9)
    plt.ion()
    plt.show()

    # read the contour shape
    contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")
    plts[0][0].set_title('Input Polygon')
    plts[0][0].fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                    alpha=0.3)
    plts[0][0].axis('equal')

    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, model.sample_num)
    plts[0][1].set_title('Uniform boundary sampling')
    plts[0][1].fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                    alpha=0.3)
    plts[0][1].axis('equal')

    # convert to polar coordinate shape
    center = model.center_point
    if center is None:
        center = getVisiblePoint(contour)
        if center is None:
            logging.error("No visible point found, need manually assign one!")
            exit(1)

    polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, model.sample_num)
    polar_contour = toEuclideanCoordAsNp(polar_poly, center[0], center[1])
    plts[1][0].set_title('Polar shape')
    plts[1][0].fill(polar_contour[:, 0], polar_contour[:, 1], "g", alpha=0.3)
    for p in polar_contour[1:-1: int(len(polar_contour) / 32)]:
        l = Line2D([center[0], p[0]], [center[1], p[1]], linewidth=1)
        plts[1][0].add_line(l)
    plts[1][0].scatter(center[0], center[1], s=10, c='b')
    plts[1][0].axis('equal')

    # generate and draw the dual shape
    driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)
    dual_shape = toEuclideanCoordAsNp(driven_gear, 0, 0)
    plts[0][2].set_title('Dual shape(Math)')
    plts[0][2].fill(dual_shape[:, 0], dual_shape[:, 1], "g", alpha=0.3)
    for p in dual_shape[1:-1: int(len(dual_shape) / 32)]:
        l = Line2D([0, p[0]], [0, p[1]], linewidth=1)
        plts[0][2].add_line(l)
    plts[0][2].scatter(0, 0, s=10, c='b')
    plts[0][2].axis('equal')
    plot_sampled_function((polar_poly, driven_gear), (phi,), debugger.get_math_debug_dir_name() if debug_mode else None,
                         100, 0.001, [(0, 0), (center_distance, 0)], (8, 8), ((-800, 1600), (-1200, 1200)))

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
        new_contour.append(( x - center[0], y - center[1]))
    # for x, y in contour:
    #     new_contour.append((y - center[1], x - center[0]))
    # BUG: I DON'T KNOW WHAT HAPPENED BUT ORIGINAL DRIVE_GEAR YOU GAVE IN THE MATH CUT AND SHAPELY CUT WERE DIFFERENT
    # new_contour = [(x, -y) for x, y in new_contour]
    drive_gear = Polygon(new_contour)
    drive_gear = drive_gear.buffer(2) # resolve invalid polygon issues
    driven_gear, cut_fig, subplot = rotate_and_cut(drive_gear, center_distance, phi, k = model.k,
                                                   debugger=debugger if debug_mode else None, replay_animation=False)
    translated_driven_gear = translate(driven_gear, center_distance)
    # final_polygon = max(translated_driven_gear, key=lambda a: a.area)
    # cutted_gear_contour = np.array(translated_driven_gear.exterior.coords)
    subplot.set_title('Dual Shape(Cut)')
    subplot.axis('equal')


    cut_fig.savefig(os.path.join(debugger.get_root_debug_dir_name(), 'cut_final.pdf'))
    fig.savefig(os.path.join(debugger.get_root_debug_dir_name(), 'shapes.pdf'))
