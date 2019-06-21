from debug_util import MyDebugger
from models import our_models
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
    plt.ion()
    fig = plt.figure()  # an empty figure with no axes
    fig.suptitle('No axes on this figure')  # Add a title so we know which it is

    figs, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
