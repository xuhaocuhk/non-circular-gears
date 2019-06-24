from models import our_models
from compute_shape_center import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
from debug_util import MyDebugger
import os
import logging
from matplotlib.lines import Line2D
from fabrication import generate_2d_obj

if __name__ == '__main__':
    debug_mode = False

    model = our_models[7]
    debugger = MyDebugger(model.name)

    # set up the plotting window
    fig, plts = plt.subplots(1, 2)
    fig.set_size_inches(16, 9)
    plt.ion()
    plt.show()

    # read the contour shape
    contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, model.sample_num)
    plts[0].set_title('Uniform boundary sampling')

    polygon = Polygon(contour)
    poly_bound = polygon.bounds
    for i in range(100):
        x_i = (poly_bound[2] - poly_bound[0]) * np.random.random_sample() + poly_bound[0]
        y_i = (poly_bound[3] - poly_bound[1]) * np.random.random_sample() + poly_bound[1]
        if polygon.contains(Point(x_i, y_i)):
            plts[0].fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                         alpha=0.3)
            plts[0].axis('equal')
            plts[0].scatter(x_i, y_i,  s=10, c='r')
            polar_poly = toExteriorPolarCoord(Point(x_i, y_i), contour, model.sample_num)
            # generate and draw the dual shape
            driven_gear, center_distance, phi = compute_dual_gear(polar_poly, k=model.k)
            dual_shape = toEuclideanCoordAsNp(driven_gear, 0, 0)
            plts[1].set_title('Dual shape(Math)')
            plts[1].fill(dual_shape[:, 0], dual_shape[:, 1], "g", alpha=0.3)
            for p in dual_shape[1:-1: int(len(dual_shape) / 32)]:
                l = Line2D([0, p[0]], [0, p[1]], linewidth=1)
                plts[1].add_line(l)
            plts[1].scatter(0, 0, s=30, c='b')
            plts[1].axis('equal')
            fig.savefig(os.path.join(debugger.get_root_debug_dir_name(), f'shapes_{i}.png'))
            plts[0].cla()
            plts[1].cla()


