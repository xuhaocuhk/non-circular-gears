
from shape_processor import *
from core.compute_dual_gear import compute_dual_gear, rotate_and_cut, _plot_polygon
from shapely.affinity import translate
from debug_util import MyDebugger
from models import Model
from core.plot_sampled_function import plot_sampled_function


def cut_gear(model: Model, debugger: MyDebugger ):
    contour = getSVGShapeAsNp(filename=f"../silhouette/{model.name}.txt")
    # convert to uniform coordinate
    contour = getUniformContourSampledShape(contour, model.sample_num)
    print(f'center={model.center_point}')

    center = model.center_point
    # get phi
    polar_poly = toExteriorPolarCoord(Point(center[0], center[1]), contour, model.sample_num)
    driven_gear, center_distance, phi = compute_dual_gear(polar_poly, 1)

    print(center_distance)
    # add teeth
    contour = addToothToContour(contour, height=model.tooth_height, tooth_num=model.tooth_num)

    # contour -= np.array(center)
    new_contour = []

    for x, y in contour:
        new_contour.append((y - center[1], x - center[0]))
    drive_gear = Polygon(new_contour)
    driven_gear = rotate_and_cut(drive_gear, center_distance, phi, debugger)
    translated_driven_gear = translate(driven_gear, center_distance)
    _plot_polygon((drive_gear, translated_driven_gear))
    plt.savefig(os.path.join(debugger.get_root_debug_dir_name(), f'cut_final.png'))
    plt.show()


if __name__ == '__main__':
    pass

