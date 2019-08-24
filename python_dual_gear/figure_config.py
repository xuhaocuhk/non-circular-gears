"""
config of colors
all colors shall be in matplotlib style
"""

figure_size = (20, 20)
axis_range = {
    'x_lim': (-3, 7),
    'y_lim': (-5, 5)
}
input_shapes = {
    'drive_face': (0.8, 0.8, 0.8),
    'drive_edge': (0.498, 0.498, 0.498),
    'driven_face': (1.0, 0.0, 0.0),
    'driven_edge': (0.498, 0.498, 0.498),
}
math_shapes = {
    key: tuple((item / 255 for item in value))
    for key, value in {
        'drive_face': (183, 208, 207),
        'drive_edge': (127, 127, 127),
        'driven_face': (161, 187, 169),
        'driven_edge': (127, 127, 127),
    }.items()
}
carve_shapes = {
    key: tuple((item / 255 for item in value))
    for key, value in {
        'drive_face': (165, 189, 225),
        'drive_edge': (127, 127, 127),
        'driven_face': (102, 160, 218),
        'driven_edge': (127, 127, 127),
    }.items()
}
