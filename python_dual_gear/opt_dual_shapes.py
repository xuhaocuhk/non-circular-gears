from models import our_models, Model
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
from debug_util import MyDebugger
from scipy.optimize import dual_annealing
import os
import logging
from matplotlib.lines import Line2D
from fabrication import generate_2d_obj
from objective_function import shape_difference_rating
import shape_factory
from typing import Union

step = 0


def obj_func(center, drive_polygon, target_shape, k, objective_sample_count=32, subplots=None, figure=None, model=None):
    drive_contour = np.array(list(drive_polygon.exterior.coords))

    dual_shape = None
    score = None
    global step

    if drive_polygon.contains(Point(center[0], center[1])):
        polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), drive_contour, model.sample_num)
        # generate and draw the dual shape
        driven_gear, center_distance, phi = compute_dual_gear(polar_poly, k=k)
        dual_shape = toCartesianCoordAsNp(driven_gear, 0, 0)
        score = shape_difference_rating(dual_shape, target_shape, objective_sample_count)
    else:
        score = 1e8

    if subplots is not None:
        subplots[0].set_title('Input shape')
        subplots[0].fill(drive_contour[:, 0], drive_contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered',
                         linewidth=3, alpha=0.3)
        subplots[0].axis('equal')
        subplots[0].scatter(center[0], center[1], s=10, c='r')

    plts = subplots  # not trying to separate drawing and calculating below
    if dual_shape is not None:
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

    for subplot in subplots:
        subplot.axis('equal')

    step = step + 1

    return score


def optimize_shapes(our_model: Model, target_model: Model, seed: int, visualization: Union[None, tuple] = None):
    # read the contour shape
    contour = shape_factory.get_shape_contour(our_model, True, None, smooth=our_model.smooth)
    target_contour = shape_factory.get_shape_contour(target_model, True, None, smooth=our_model.smooth)

    if visualization is not None:
        figure, subplots = visualization
    else:
        figure, subplots = None, None
    polygon = Polygon(contour)
    poly_bound = polygon.bounds

    lb = [poly_bound[0], poly_bound[1]]
    ub = [poly_bound[2], poly_bound[3]]

    return dual_annealing(obj_func, args=(polygon, target_contour, 1, 32, subplots, figure, our_model),
                          bounds=list(zip(lb, ub)), seed=seed, maxiter=100)


if __name__ == '__main__':
    debug_mode = False
    debugger = MyDebugger(['circle', 'square'])

    # set up the plotting window
    fig, plts = plt.subplots(1, 3)
    fig.set_size_inches(16, 9)
    plt.ion()
    plt.show()

    ret = optimize_shapes(our_models[0], our_models[-1], 3, (fig, plts))

    print(f"global minimum: xmin = {ret.x}, f(xmin) = {ret.fun}")
