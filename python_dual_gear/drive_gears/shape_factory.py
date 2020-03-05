from drive_gears.shape_processor import *
from drive_gears.models import Model
from drive_gears.standard_shapes import std_shapes, generate_std_shapes
from plot.plot_util import plot_cartesian_shape
from matplotlib.axes import Axes
from typing import Union, Iterable, Optional
import os
import cv2


def get_shape_contour(model: Model, uniform: bool, plots: Union[Iterable[Axes], None], smooth=0,
                      face_color=None, edge_color=None):
    contour = None
    if model.name in std_shapes:
        contour = generate_std_shapes(model.name, model.sample_num, model.center_point)
    else:
        # read the contour shape
        # extract the source if hinted
        if '(' in model.name:
            sub_folder, model_name = model.name.split(')')
            sub_folder = sub_folder[1:]
            silhouette_file = find_silhouette_file(model_name, os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../silhouette/' + sub_folder)))
        else:
            silhouette_file = find_silhouette_file(model.name)
        if silhouette_file is None:
            raise FileNotFoundError(f'silhouette {model.name} not found!')
        contour = getSVGShapeAsNp(filename=silhouette_file)

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


def find_silhouette_file(model_name: str, base_path: Optional[str] = None):
    base_path = base_path or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../silhouette'))
    target_file = os.path.join(base_path, model_name + '.txt')
    if os.path.isfile(target_file):
        return target_file  # found file
    else:
        dirs = [directory for directory in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, directory))]
        for path in dirs:
            result = find_silhouette_file(model_name, os.path.join(base_path, path))
            if result is not None:
                return result
        else:
            return None


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
    if len(contour.shape) == 1:
        print("error: " + output_path)
    else:
        with open(output_path, 'w') as file:
            print(*['{0} {1}'.format(x, y) for x, y in contour], sep=',', file=file)


def transform_all_binary_images(root_path):
    if os.path.isdir(root_path):
        # process the whole directory
        files = os.listdir(root_path)
        files = [file for file in files if file[0] != '.' and file[:2] != '__']
        for file in files:
            transform_all_binary_images(os.path.join(root_path, file))
    elif os.path.isfile(root_path):
        target_contour_name = root_path[:-4] + '.txt'
        if '.txt' not in root_path and not os.path.exists(target_contour_name):
            export_contour_as_text(target_contour_name, read_binary_image(root_path))
    else:
        raise FileNotFoundError('Invalid Filename')


if __name__ == '__main__':
    transform_all_binary_images(os.path.abspath(os.path.join(os.path.dirname(__file__), '../silhouette/')))
