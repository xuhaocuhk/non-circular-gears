from shape_processor import *
from models import Model
from drive_gears.generate_standard_shapes import std_shapes, generate_std_shapes
from plot.plot_util import plot_cartesian_shape
from matplotlib.axes import Axes
from typing import Union, Iterable
import os
import cv2


def get_shape_contour(model: Model, uniform: bool, plots: Union[Iterable[Axes], None], smooth=0,
                      face_color=None, edge_color=None):
    contour = None
    if model.name in std_shapes:
        contour = generate_std_shapes(model.name, model.sample_num, model.center_point)
    else:
        # read the contour shape
        contour = getSVGShapeAsNp(filename=os.path.join(os.path.dirname(__file__), f"../silhouette/{model.name}.txt"))

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
        contour = uniform_and_smooth(contour, model)

        if plots is not None:
            plot_cartesian_shape(subplots[1], "Uniform boundary sampling", contour)

    return contour


def uniform_and_smooth(contour, model):
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, model.sample_num, False)
    # spline uniform
    if not model.smooth == 0:
        contour = getUniformContourSampledShape(contour[::model.smooth], model.sample_num, True)
    return contour


def read_binary_image(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return np.squeeze(cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])


def export_contour_as_text(output_path, contour):
    with open(output_path, 'w') as file:
        print(*['{0} {1}'.format(x, y) for x, y in contour], sep=',', file=file)


if __name__ == '__main__':
    contour = read_binary_image(r'C:\Projects\gears\binary_images\animal_fly\4dove.png')
    export_contour_as_text('test.txt', contour)
