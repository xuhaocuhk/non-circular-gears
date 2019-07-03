from shape_processor import *
from models import Model
from debug_util import MyDebugger
from drive_gears.generate_standard_shapes import std_shapes, generate_std_shapes
from plot.plot_util import plot_cartesian_shape

def getShapeContour(model: Model, uniform_contour: bool, plts):
    contour = None
    if model.name in std_shapes:
        contour = generate_std_shapes(model.name, model.sample_num, model.center_point)
    else:
        # read the contour shape
        contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")
        plot_cartesian_shape(plts[0][0], "Input Shape", contour)

    # shape normalization


    # unifrom vertex on contour
    if uniform_contour:
        # convert to uniform coordinate
        contour = getUniformContourSampledShape(contour, model.sample_num)

        plot_cartesian_shape(plts[0][1], "Uniform boundary sampling", contour)


    return contour