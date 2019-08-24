from matplotlib.lines import Line2D
from shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
import matplotlib.pyplot as plt
import numpy as np
import figure_config


# set up the plotting window
def init_plot():
    fig, plts = plt.subplots(3, 3)
    fig.set_size_inches(16, 9)
    plt.ion()
    plt.show()
    return fig, plts


def plot_cartesian_shape(ax, title, contour, face_color=None, edge_color=None):
    ax.set_title(title)
    ax.fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon' if face_color is None else face_color,
            edgecolor='orangered' if edge_color is None else edge_color, linewidth=3, alpha=0.3)
    ax.axis('equal')


def plot_polar_shape(ax, title, polar_contour, center, sample_num):
    cartesian_contour = toCartesianCoordAsNp(polar_contour, center[0], center[1])
    ax.set_title(title)
    ax.fill(cartesian_contour[:, 0], cartesian_contour[:, 1], "g", alpha=0.3)
    for p in cartesian_contour[1:-1: int(len(cartesian_contour) / 32)]:
        l = Line2D([center[0], p[0]], [center[1], p[1]], linewidth=1)
        ax.add_line(l)
    ax.scatter(center[0], center[1], s=10, c='b')
    ax.axis('equal')


def plot_contour_and_save(contour: np.ndarray, file_path: str, face_color=None, edge_color=None, center=None):
    fig = plt.figure(figsize=figure_config.figure_size)
    plt.fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon' if face_color is None else face_color,
             edgecolor='orangered' if edge_color is None else edge_color, linewidth=3, alpha=0.3)
    plt.axis(figure_config.axis_range['x_lim'] + figure_config.axis_range['y_lim'])
    # scatter center if given
    if center is not None:
        center_circle = plt.Circle(center, figure_config.scatter_point['size'],
                                   facecolor=figure_config.scatter_point['color'],
                                   edgecolor=figure_config.scatter_point['edge'])
        fig.gca().add_artist(center_circle)
    plt.axis('off')
    plt.savefig(file_path)
    plt.close(fig)


if __name__ == '__main__':
    from debug_util import MyDebugger

    debugger = MyDebugger('test')
    contour = np.array([
        (0, 0),
        (2, 3),
        (3, -2)
    ])
    plot_contour_and_save(contour, debugger.file_path('test.png'), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
