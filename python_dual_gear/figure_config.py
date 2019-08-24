"""
config of colors
all colors shall be in Qt style
"""

figure_size = (2000, 2000)
axis_range = {
    'x_lim': (-1.5, 2.5),
    'y_lim': (-2, 2)
}
scatter_point = {
    'size': 0.05,
    'color': (1.0, 1.0, 1.0, 1.0),
    'edge': (0.498, 0.498, 0.498, 1.0)
}
input_shapes = {
    'drive_face': (0.8, 0.8, 0.8, 1),
    'drive_edge': (0.498, 0.498, 0.498, 1),
    'driven_face': (1.0, 0.0, 0.0, 1),
    'driven_edge': (0.498, 0.498, 0.498, 1),
}
math_shapes = {
    key: tuple((item / 255 for item in value)) + (1,)
    for key, value in {
        'drive_face': (183, 208, 207),
        'drive_edge': (127, 127, 127),
        'driven_face': (161, 187, 169),
        'driven_edge': (127, 127, 127),
    }.items()
}
carve_shapes = {
    key: tuple((item / 255 for item in value)) + (1,)
    for key, value in {
        'drive_face': (165, 189, 225),
        'drive_edge': (127, 127, 127),
        'driven_face': (102, 160, 218),
        'driven_edge': (127, 127, 127),
    }.items()
}
