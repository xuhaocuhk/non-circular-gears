from models import our_models
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
from debug_util import MyDebugger
from scipy.optimize import dual_annealing
import os
import logging
from matplotlib.lines import Line2D
from fabrication import generate_2d_obj


# TODO: to be replaced by professor FU(sample FU)'s version
def polygon_compare(contour, target_contour):
    x_range = max(contour[:, 0]) - min(contour[:, 0])
    y_range = max(contour[:, 1]) - min(contour[:, 1])
    target_x_range = max(target_contour[:, 0]) - min(target_contour[:, 0])
    target_y_range = max(target_contour[:, 1]) - min(target_contour[:, 1])
    return ((x_range / y_range) - (target_x_range / target_y_range)) ** 2


'''
args[0]: contour1: Polygon
args[1]: contour2: ndarray of shape(n,2)
args[2]: k
'''

step = 0


def obj_func(center, *args):
    drive_polygon = args[0]
    drive_contour = np.array(list(drive_polygon.exterior.coords))
    target_shape = args[1]
    k = args[2]
    plts = args[3]
    fig = args[4]

    dual_shape = None
    score = None
    global step

    if polygon.contains(Point(center[0], center[1])):
        polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), drive_contour, model.sample_num)
        # generate and draw the dual shape
        driven_gear, center_distance, phi = compute_dual_gear(polar_poly, k=k)
        dual_shape = toCartesianCoordAsNp(driven_gear, 0, 0)
        score = polygon_compare(dual_shape, target_shape)
    else:
        score = 1e8

    plts[0].set_title('Input shape')
    plts[0].fill(drive_contour[:, 0], drive_contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered',
                 linewidth=3,
                 alpha=0.3)
    plts[0].axis('equal')
    plts[0].scatter(center[0], center[1], s=10, c='r')
    if not dual_shape is None:
        plts[1].set_title('Dual shape(Math)')
        plts[1].fill(dual_shape[:, 0], dual_shape[:, 1], "g", alpha=0.3)
        for p in dual_shape[1:-1: int(len(dual_shape) / 32)]:
            l = Line2D([0, p[0]], [0, p[1]], linewidth=1)
            plts[1].add_line(l)
        plts[1].scatter(0, 0, s=30, c='b')
        plts[1].axis('equal')
    plts[1].text(0, 0, str(score), ha='left', rotation=0, wrap=True)
    # draw target shape
    plts[2].set_title('Target shape')
    plts[2].fill(target_shape[:, 0], target_shape[:, 1], "g", alpha=0.3)
    plts[2].axis('equal')
    fig.savefig(os.path.join(debugger.get_root_debug_dir_name(), f'shapes_{step}_{score}.png'))
    plt.pause(0.0001)
    plts[0].cla()
    plts[1].cla()
    plts[2].cla()
    step = step + 1

    return score


if __name__ == '__main__':
    debug_mode = False

    model = our_models[7]
    target_model = our_models[4]
    debugger = MyDebugger(model.name)

    # set up the plotting window
    fig, plts = plt.subplots(1, 3)
    fig.set_size_inches(16, 7)
    plt.ion()
    plt.show()

    # read the contour shape
    contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")
    target_contour = getSVGShapeAsNp(filename=f"../silhouette/{target_model.name}.txt")
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, model.sample_num)
    target_contour = getUniformContourSampledShape(target_contour, target_model.sample_num)
    # plts[0].set_title('Uniform boundary sampling')

    polygon = Polygon(contour)
    poly_bound = polygon.bounds

    lb = [poly_bound[0], poly_bound[1]]
    ub = [poly_bound[2], poly_bound[3]]
    ret = dual_annealing(obj_func, args=(polygon, target_contour, 1, plts, fig), bounds=list(zip(lb, ub)), seed=3,
                         maxiter=200)
    print(f"global minimum: xmin = {ret.x}, f(xmin) = {ret.fun}")
