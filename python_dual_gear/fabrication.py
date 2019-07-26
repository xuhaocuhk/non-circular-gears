"""
functions related to 3D modelling and fabrication
"""
from debug_util import MyDebugger
import numpy as np
import openmesh
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
    mesh = openmesh.TriMesh()
    lower_plane = [mesh.add_vertex((np.array(*point, 0) for point in contour))]  # TODO: make this more graceful
    upper_plane = [mesh.add_vertex((np.array(*point, thickness) for point in contour))]
    contour_poly = Polygon(contour)
    point_to_vertex = {
        tuple(point): (lower_vertex, upper_vertex)
        for point, lower_vertex, upper_vertex in zip(contour, lower_plane, upper_plane)
    }
    triangles = triangulate(contour_poly)
    for triangle in triangles:
        *points, _ = triangle
        mesh.add_face(zip(*[point_to_vertex[point] for point in points]))  # TODO: is this correct?


def generate_printable_spline(debugger, contour1, contour2, center_dist, target_dist=100):
    scale = target_dist / center_dist
    contour1_scale = [[point[0] * scale, point[1] * scale] for point in contour1]
    contour2_scale = [[point[0] * scale, point[1] * scale] for point in contour2]
    generate_2d_obj(debugger, 'print_drive.obj', contour1_scale)
    generate_2d_obj(debugger, 'print_driven.obj', contour2_scale)


if __name__ == '__main__':
    from plot.plot_sampled_function import polar_to_rectangular
    from drive_gears.ellipse_gear import generate_gear

    generate_2d_obj('test.obj', polar_to_rectangular(generate_gear(1024), None))
