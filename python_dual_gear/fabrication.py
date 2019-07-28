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
    contour = np.append(contour, contour[:1], axis=0)  # duplicate first point to complete a cycle
    lower_plane = [mesh.add_vertex((np.append(point, 0))) for point in contour]
    upper_plane = [mesh.add_vertex((np.append(point, thickness))) for point in contour]
    contour_poly = Polygon(contour)
    point_to_vertex = {
        tuple(point): (lower_vertex, upper_vertex)
        for point, lower_vertex, upper_vertex in zip(contour, lower_plane, upper_plane)
    }
    triangles = triangulate(contour_poly)
    lower_face = []
    upper_face = []
    for triangle in triangles:
        *points, _ = triangle.exterior.coords
        face_1, face_2 = zip(*[point_to_vertex[point] for point in points])
        lower_face.append(mesh.add_face(face_1))
        upper_face.append(mesh.add_face(face_2))

    openmesh.write_mesh(destination, mesh)


def generate_printable_spline(debugger, contour1, contour2, center_dist, target_dist=100):
    scale = target_dist / center_dist
    contour1_scale = [[point[0] * scale, point[1] * scale] for point in contour1]
    contour2_scale = [[point[0] * scale, point[1] * scale] for point in contour2]
    generate_2d_obj(debugger, 'print_drive.obj', contour1_scale)
    generate_2d_obj(debugger, 'print_driven.obj', contour2_scale)


if __name__ == '__main__':
    square_contour = np.array([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    generate_3d_mesh(MyDebugger('test'), 'output.obj', square_contour, 0.5)
