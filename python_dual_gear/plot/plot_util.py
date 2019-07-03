from shapely.geometry import Point
from matplotlib.lines import Line2D
from shape_processor import toExteriorPolarCoord, toCartesianCoordAsNp
import matplotlib.pyplot as plt

# set up the plotting window
def init_plot():
    fig, plts = plt.subplots(2, 3)
    fig.set_size_inches(16, 9)
    plt.ion()
    plt.show()
    return fig, plts

def plot_cartesian_shape(ax, title, contour):
    ax.set_title(title)
    ax.fill(contour[:, 0], contour[:, 1], "g", facecolor='lightsalmon', edgecolor='orangered', linewidth=3,
                    alpha=0.3)

def plot_polar_shape(ax, title, polar_contour, center, sample_num):
    cartesian_contour = toCartesianCoordAsNp(polar_contour, center[0], center[1])
    ax.set_title(title)
    ax.fill(cartesian_contour[:, 0], cartesian_contour[:, 1], "g", alpha=0.3)
    for p in cartesian_contour[1:-1: int(len(cartesian_contour) / 32)]:
        l = Line2D([center[0], p[0]], [center[1], p[1]], linewidth=1)
        ax.add_line(l)
    ax.scatter(center[0], center[1], s=10, c='b')
    ax.axis('equal')

