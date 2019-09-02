"""
functions related to 3D modelling and fabrication
"""
from debug_util import MyDebugger
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.ops import triangulate
from typing import Tuple, Union, Optional
import os
from core.optimize_dual_shapes import clockwise_orientation
import math


def read_2d_obj(filename):
    """
    Given a file path, read the 2d obj file and return an np.ndarray
    :param:file_path: the file path of the 2d obj file
    :return: return an ndarray
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines:
            str_list = line.split()
            if str_list[0] == 'v':
                point = Point(float(str_list[1]), float(str_list[2]))
                points.append(point)
    result = [(point.x, point.y) for point in points]
    result = np.array(result)
    result = clockwise_orientation(result)
    return result


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
    :return: None
    """
    if filename[-4:] != '.obj':
        filename = filename + '.obj'
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
            triangle_bound = LineString(triangle.exterior)
            if not triangle_bound.within(contour_poly):
                continue
            *points, _ = triangle.exterior.coords
            face_1, face_2 = zip(*[point_to_vertex[point] for point in points])
            for face in (face_1[::-1], face_2):
                print('f ' + ' '.join([str(i) for i in face]), file=obj_file)
        for index, point in enumerate(contour):
            lower_point, upper_point = point_to_vertex[tuple(point)]
            lower_prev, upper_prev = point_to_vertex[tuple(contour[index - 1])]
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_point, upper_point)]), file=obj_file)
            print('f ' + ' '.join([str(point) for point in (upper_prev, lower_prev, lower_point)]), file=obj_file)


def generate_3d_mesh_hole(debugger: Union[MyDebugger, str], filename: str, contour: np.ndarray, interiors: np.ndarray,
                          thickness=7.76):
    """
    this is the function to generate an obj file of a polygon with an inner hole (serves as the axis hole)
    :param debugger: the debugger to provide directory for obj to be stored
    :param filename: filename(excluding the extension)
    :param contour:  the contour to create 3d object with (exterior of a polygon)
    :param interiors: the interior pts of the polygon
    :param thickness: the thickness of the object
    :return: None
    """
    if isinstance(debugger, MyDebugger):
        destination = debugger.file_path(filename)
    else:
        assert os.path.isdir(debugger)
        destination = os.path.join(debugger, filename)
    with open(destination, 'w') as obj_file:
        point_to_vertex = {}
        for index, point in enumerate(contour):
            point_to_vertex[tuple(point)] = (index * 2 + 1, index * 2 + 2)
            print(f'v {point[0]}, {point[1]} 0', file=obj_file)
            print(f'v {point[0]}, {point[1]} {thickness}', file=obj_file)
        for index, point in enumerate(interiors):
            index += len(contour)
            point_to_vertex[tuple(point)] = (index * 2 + 1, index * 2 + 2)
            print(f'v {point[0]}, {point[1]} 0', file=obj_file)
            print(f'v {point[0]}, {point[1]} {thickness}', file=obj_file)

        poly = Polygon(contour, [interiors])
        triangles = triangulate(poly)
        for triangle in triangles:
            triangle_bound = LineString(triangle.exterior)
            if not triangle_bound.within(poly):
                continue
            *points, _ = triangle.exterior.coords
            face_1, face_2 = zip(*[point_to_vertex[point] for point in points])
            for face in (face_1[::-1], face_2):
                print('f ' + ' '.join([str(i) for i in face]), file=obj_file)
        for index, point in enumerate(contour):
            lower_point, upper_point = point_to_vertex[tuple(point)]
            lower_prev, upper_prev = point_to_vertex[tuple(contour[index - 1])]
            print('f ' + ' '.join([str(point) for point in (upper_point, lower_point, upper_prev)]), file=obj_file)
            print('f ' + ' '.join([str(point) for point in (lower_point, lower_prev, upper_prev)]), file=obj_file)
        for index, point in enumerate(interiors):
            lower_point, upper_point = point_to_vertex[tuple(point)]
            lower_prev, upper_prev = point_to_vertex[tuple(interiors[index - 1])]
            print('f ' + ' '.join([str(point) for point in (upper_point, lower_point, upper_prev)]), file=obj_file)
            print('f ' + ' '.join([str(point) for point in (lower_point, lower_prev, upper_prev)]), file=obj_file)


def generate_printable_spline(debugger, contour1, contour2, center_dist, target_dist=100):
    scale = target_dist / center_dist
    contour1_scale = [[point[0] * scale, point[1] * scale] for point in contour1]
    contour2_scale = [[point[0] * scale, point[1] * scale] for point in contour2]
    generate_2d_obj(debugger, 'print_drive.obj', contour1_scale)
    generate_2d_obj(debugger, 'print_driven.obj', contour2_scale)


def draw_cross(axis):
    """
    Return an array of vertices of a 2d cross axis.
    :param axis: the axis coordinate
    :return cross: an array of vertices of a 2d cross axis.
    """
    x = axis.x
    y = axis.y
    tolerance = 0.3
    cross_contour = [(x + 1.20, y + 1.20),
                     (x + 0.90, y + 1.20),
                     (x + 0.90, y + 2.40),
                     (x - 0.90, y + 2.40),
                     (x - 0.90, y + 1.20),
                     (x - 1.20, y + 1.20),
                     (x - 1.20, y + 0.90),
                     (x - 2.40, y + 0.90),
                     (x - 2.40, y - 0.90),
                     (x - 1.20, y - 0.90),
                     (x - 1.20, y - 1.20),
                     (x - 0.90, y - 1.20),
                     (x - 0.90, y - 2.40),
                     (x + 0.90, y - 2.40),
                     (x + 0.90, y - 1.20),
                     (x + 1.20, y - 1.20),
                     (x + 1.20, y - 0.90),
                     (x + 2.40, y - 0.90),
                     (x + 2.40, y + 0.90),
                     (x + 1.20, y + 0.90)]
    result = []
    for vertex_tuple in cross_contour:
        vertex_list = list(vertex_tuple)
        if vertex_list[0] > 0:
            vertex_list[0] += tolerance
        else:
            vertex_list[0] -= tolerance
        if vertex_list[1] > 0:
            vertex_list[1] += tolerance
        else:
            vertex_list[1] -= tolerance
        result.append(tuple(vertex_list))
    return result


def generate_3D_with_axles(distance: float, filename_drive: str, filename_driven: str, drive_axis: Tuple[float, float],
                           driven_axis: Tuple[float, float], debugger: Optional[MyDebugger], thickness=7.76):
    """
    :param distance: distance between axes of two gears
    :param filename_drive: file name of the drive gear
    :param filename_driven: file name of the driven gear
    :param debugger: None to save in the same directory as the input
    :return:
    """
    filename_drive = os.path.abspath(filename_drive)
    filename_driven = os.path.abspath(filename_driven)
    assert os.path.isfile(filename_drive)
    assert os.path.isfile(filename_driven)
    exterior_drive = read_2d_obj(filename_drive)
    exterior_driven = read_2d_obj(filename_driven)
    scaling_ratio = distance * 7.97 / (math.fabs(drive_axis[0] - driven_axis[0]))
    exterior_drive_scale = [(scaling_ratio * drive_point[0], scaling_ratio * drive_point[1]) for drive_point in
                            exterior_drive]
    exterior_driven_scale = [(scaling_ratio * driven_point[0], scaling_ratio * driven_point[1]) for driven_point in
                             exterior_driven]
    drive_eroded = Polygon(exterior_drive_scale).buffer(-0.3)
    driven_eroded = Polygon(exterior_driven_scale).buffer(-0.3)
    exterior_drive_eroded = drive_eroded.exterior.coords
    exterior_driven_eroded = driven_eroded.exterior.coords
    drive_axis_scale = Point(drive_axis[0] * scaling_ratio, drive_axis[1] * scaling_ratio)
    driven_axis_scale = Point(driven_axis[0] * scaling_ratio, driven_axis[1] * scaling_ratio)
    interior_drive = draw_cross(drive_axis_scale)
    interior_driven = draw_cross(driven_axis_scale)
    if debugger is None:
        destination_directory = os.path.dirname(filename_drive)
        debugger = destination_directory
    generate_3d_mesh_hole(debugger, 'drive_gear_mesh.obj', np.array(exterior_drive_eroded), np.array(interior_drive),
                          thickness)
    generate_3d_mesh_hole(debugger, 'driven_gear_mesh.obj', np.array(exterior_driven_eroded), np.array(interior_driven),
                          thickness)


if __name__ == '__main__':
    import math

    square_contour = np.array(
        [(6 * math.cos(theta), 6 * math.sin(theta)) for theta in np.linspace(0, 2 * math.pi, 1024, endpoint=False)])
    # generate_3d_mesh(MyDebugger('test'), 'output.obj', square_contour, 0.5)

    # cross_contour = draw_cross(Point(0, 0))
    # polygon_ext = [(-5, -5), (5, -5), (5, 5), (-5, 5)]

    # generate_3d_mesh_hole(MyDebugger('test'), 'output.obj', square_contour, cross_contour, 2)
    filename_drive = './debug/drive_2d.obj'
    filename_driven = './debug/driven_2d.obj'
    drive_axis = (0, 0)
    driven_axis = (0.8251464417682275, 0)
    generate_3D_with_axles(8, filename_drive, filename_driven, drive_axis, driven_axis, None, 7.76)
