from shape_processor import *
from models import Model
from drive_gears.generate_standard_shapes import std_shapes, generate_std_shapes
from plot.plot_util import plot_cartesian_shape
from matplotlib.axes import Axes
from typing import Union, Iterable


def get_shape_contour(model: Model, uniform: bool, plots: Union[Iterable[Axes], None], smooth=0,
                      face_color=None, edge_color=None):
    contour = None
    if model.name in std_shapes:
        contour = generate_std_shapes(model.name, model.sample_num, model.center_point)
    else:
        # read the contour shape
        contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")

    # shape normalization
    poly_bound = Polygon(contour).bounds
    max_axis = max((poly_bound[2] - poly_bound[0]), (poly_bound[3] - poly_bound[1]))
    contour = contour / max_axis

    subplots = []  # dummy value to suppress dummy warning
    if plots is not None:
        subplots = list(plots)
        plot_cartesian_shape(subplots[0], "Input Shape", contour, face_color, edge_color)

    # uniform vertex on contour
    if uniform:
        # convert to uniform coordinate
        contour = getUniformContourSampledShape(contour, model.sample_num, False)
        if not smooth == 0:
            contour = getUniformContourSampledShape(contour[::smooth], model.sample_num, True)

        if plots is not None:
            plot_cartesian_shape(subplots[1], "Uniform boundary sampling", contour)

    return contour
