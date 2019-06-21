from debug_util import MyDebugger
from models import our_models
from examples import pick_center_point, cut_gear
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
    fig, plts = plt.subplots(2, 2)
    l = Line2D([0, 1], [0, 1])
    plts[0][0].add_line(l)
    plt.show()