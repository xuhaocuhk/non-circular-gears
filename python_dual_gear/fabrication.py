"""
functions related to 3D modelling and fabrication
"""
from debug_util import MyDebugger
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate


def generate_2d_obj(debugger, filename, points):
    filename = debugger.file_path(filename)
    with open(filename, 'w') as file:
        for point in points:
            x, y = point
            print(f'v {x} {y} 0', file=file)
        print('o Spine', file=file)
        print('g Segment1', file=file)
        print('l', end=' ', file=file)
        for i in range(len(points)):
            print(i + 1, end=' ', file=file)
        print(1, file=file)


def generate_3d_mesh(debugger: MyDebugger, filename: str, contour: np.ndarray, thickness: float):
    """
    generate a 3D mesh of the given contour with the given thickness
    :param debugger: the debugger to provide directory for obj to be stored
    :param filename: filename (excluding the extension)
    :param contour: the contour to create 3d object with
    :param thickness: the thickness of 3d object mesh
    :return:
    """
    destination = debugger.file_path(filename)
    with open(destination, 'w') as obj_file:
        point_to_vertex = {}
        for index, point in enumerate(contour):
            point_to_vertex[tuple(point)] = (index * 2 + 1, index * 2 + 2)
            print(f'v {point[0]} {point[1]} 0', file=obj_file)
            print(f'v {point[0]} {point[1]} {thickness}', file=obj_file)

        contour_poly = Polygon(contour)
        triangles = triangulate(contour_poly)
        for triangle in triangles:
            *points, _ = triangle.exterior.coords
            face_1, face_2 = zip(*[point_to_vertex[point] for point in points])
            for face in (face_1[::-1], face_2):
                print('f ' + ' '.join([str(i) for i in face]), file=obj_file)
        for index, point in enumerate(contour):
            lower_point, upper_point = point_to_vertex[tuple(point)]
            lower_prev, upper_prev = point_to_vertex[tuple(contour[index - 1])]
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_point, upper_point)]), file=obj_file)
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_prev, lower_point)]), file=obj_file)


def generate_printable_spline(debugger, contour1, contour2, center_dist, target_dist=100):
    scale = target_dist / center_dist
    contour1_scale = [[point[0] * scale, point[1] * scale] for point in contour1]
    contour2_scale = [[point[0] * scale, point[1] * scale] for point in contour2]
    generate_2d_obj(debugger, 'print_drive.obj', contour1_scale)
    generate_2d_obj(debugger, 'print_driven.obj', contour2_scale)


if __name__ == '__main__':
    import math

    square_contour = np.array(
        [(3 * math.cos(theta), 2 * math.sin(theta)) for theta in np.linspace(0, 2 * math.pi, 1024, endpoint=False)])
    generate_3d_mesh(MyDebugger('test'), 'output.obj', square_contour, 0.5)
