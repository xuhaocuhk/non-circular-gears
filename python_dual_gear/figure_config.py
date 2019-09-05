"""
config of colors
all colors shall be in Qt style
"""

figure_size = (2000, 2000)
figure_translation = (1.4, 2)
figure_scale = 500  # translation done before scaling
edge_width = 5  # in pixels
axis_range = {
    'x_lim': (-1.5, 2.5),
    'y_lim': (-2, 2)
}
scatter_point = {
    'size': 0.05,  # legacy part in some files
    'radius': 10,
    'edge_width': 3,
    'color': (255, 255, 255),
    'edge': (127, 127, 127)
}
input_shapes = {
    'drive_face': (204, 204, 204),
    'drive_edge': (127, 127, 127),
    'driven_face': (204, 204, 204),
    'driven_edge': (127, 127, 127),
}
math_shapes = {
    'drive_face': (183, 208, 207),
    'drive_edge': (127, 127, 127),
    'driven_face': (161, 187, 169),
    'driven_edge': (127, 127, 127),
}
carve_shapes = {
    'drive_face': (165, 189, 225),
    'drive_edge': (127, 127, 127),
    'driven_face': (102, 160, 218),
    'driven_edge': (127, 127, 127),
}
