from shape_processor import *
from models import Model
from debug_util import MyDebugger
from drive_gears.generate_standard_shapes import std_shapes, generate_std_shapes

def getShapeContour(model: Model, do_uniform: bool, plts):
    debugger = MyDebugger(model.name)
    contour = None
    if model.name in std_shapes:
        contour = generate_std_shapes(model.name, model.sample_num, model.center_point)
    else:
        # read the contour shape
        contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")

    plts[0][0].set_title('Input Polygon')
    plts[0][0].fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                    alpha=0.3)
    plts[0][0].axis('equal')

    if do_uniform:
        # convert to uniform coordinate
        contour = getUniformContourSampledShape(contour, model.sample_num)
        plts[0][1].set_title('Uniform boundary sampling')
        plts[0][1].fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                        alpha=0.3)
        plts[0][1].axis('equal')

    return contour, debugger