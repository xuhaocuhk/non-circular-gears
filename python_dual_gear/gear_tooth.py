import math

import numpy as np
from shapely.geometry import Polygon

from util_functions import counterclockwise_orientation
from drive_gears.shape_processor import getNormals


def teeth_straight(x: float, height: float, width: float):
    assert 0 <= x <= 1
    if x < width / 7 * 2:
        y = height * (x / (width / 7 * 2))
    elif x < width / 7 * 5:
        y = height
    elif x < width:
        y = height * (width - x) / (width / 7 * 2)
    else:
        y = 0.0
    return y - height / 2


def teeth_sine(x: float, height: float, width: float):
    assert 0 <= x <= 1
    if x < width:
        return height * math.sin(x * 2 * math.pi)
    else:
        return 0


def teeth_involute(x: float, height: float, width: float):
    assert 0 <= x <= 1
    assert 0 < width < 1
    fake_height = height / (-(2 / 3 - 1) ** 2 + 1)
    if x < width / 3:
        y = fake_height * (-(2 / width * x - 1) ** 2 + 1)
    elif x < width / 3 * 2:
        y = height
    elif x < width:
        y = fake_height * (-(2 / width * x - 1) ** 2 + 1)
    else:
        y = 0
    return y - height / 2


def teeth_involute_sin(x: float, height: float, width: float):
    assert 0 <= x <= 1
    assert 0 < width <= 1

    def sin_involute(x: float):
        assert 0 <= x <= 1
        fake_height = height / math.sin(math.pi / 3)
        if x < 1 / 6 or (x > 2 / 6 and x < 3.8 / 6) or x > 5.2 / 6:
            return math.sin(x * 2 * math.pi) * fake_height
        elif x <= 2 / 6:
            return height
        else:
            return -height

    y = sin_involute(x)

    return y


# return the average driving ratio of the given range
def sample_avg(start, end, polar_contour, center_dist):
    return np.average([d / (center_dist - d) for d in polar_contour[start:end]])


def normal_mid(start, end, normals):
    return normals[(start + end) // 2]


def point_mid(start, end, contour, center):
    return contour[(start + end) // 2] - center


# return a value in [0,1] the map the teeth height
def get_value_on_tooth_domain(i: int, _tooth_samples):
    assert i < _tooth_samples[-1]
    idx = np.argmax(_tooth_samples > i)
    if idx == 0:
        value = i / _tooth_samples[0]
    else:
        value = (i - _tooth_samples[idx - 1]) / (_tooth_samples[idx] - _tooth_samples[idx - 1])
    assert 0 <= value <= 1
    return value


def get_teeth_idx(i: int, _tooth_samples):
    assert i < _tooth_samples[-1]
    return np.argmax(_tooth_samples > i)


def add_teeth(center, center_distance, debugger, drive, drive_model, plotter):
    drive = counterclockwise_orientation(drive)
    normals = getNormals(drive, None, center, normal_filter=True)
    drive = addToothToContour(drive, center, center_distance, normals, height=drive_model.tooth_height,
                              tooth_num=drive_model.tooth_num,
                              plt_axis=None, consider_driving_torque=False,
                              consider_driving_continue=False)
    plotter.draw_contours(debugger.file_path('drive_with_teeth_before.png'), [('input_driven', drive)], None)

    drive = Polygon(drive).buffer(0).simplify(0.000)
    if drive.geom_type == 'MultiPolygon':
        drive = max(drive, key=lambda a: a.area)
    drive = np.array(drive.exterior.coords)
    plotter.draw_contours(debugger.file_path('drive_with_teeth.png'), [('input_driven', drive)], None)
    return drive


def addToothToContour(contour: np.array, center, center_dist, normals, height: int, tooth_num: int, plt_axis,
                      consider_driving_torque=False, consider_driving_continue=False):
    n = len(contour)
    assert n % tooth_num == 0
    samplenum_per_teeth = n / tooth_num

    polar_contour = [np.linalg.norm(p - center) for p in contour]

    tooth_samples = np.full(tooth_num, samplenum_per_teeth, dtype=np.int_)  # how many points compose a tooth
    tooth_samples = np.cumsum(tooth_samples)
    heights = np.full(tooth_num, height)  # what's the height of each tooth

    if consider_driving_torque:
        for i in range(10):
            tooth_samples = np.insert(tooth_samples, 0, 0)
            driving_ratios = np.array(
                [gear_tooth.sample_avg(tooth_samples[j], tooth_samples[j + 1], polar_contour, center_dist) for j in
                 range(tooth_num)],
                dtype=np.float_)
            driving_ratios = np.array(list(map(lambda x: x ** (1 / 4), driving_ratios)))  # to discriminish the gaps
            driving_ratios = driving_ratios / np.sum(
                driving_ratios) * n  # resample the samples according to the driving ratio of current tooth
            re_indexing = np.cumsum(driving_ratios)
            tooth_samples = np.round(re_indexing).astype('int32')

    if consider_driving_continue:
        tooth_widths = np.diff(np.insert(tooth_samples, 0, 0))  # again, how many points compose a tooth
        for j in range(tooth_num):
            # front zero padding
            zero_front_tooth_samples = np.insert(tooth_samples, 0, 0)

            curr_normal = normal_mid(zero_front_tooth_samples[j], zero_front_tooth_samples[j + 1], normals)
            curr_center_direction = point_mid(zero_front_tooth_samples[j], zero_front_tooth_samples[j + 1],
                                              contour, center)
            sin_theta = np.cross(curr_normal, curr_center_direction) / (
                    np.linalg.norm(curr_normal) * np.linalg.norm(curr_center_direction))
            if sin_theta < 0:
                pass  # heights[j] = height
            else:
                perimeter = Polygon(contour).length
                heights[j] = (perimeter * tooth_widths[j] / n) * (
                        sin_theta / math.sqrt(1 - sin_theta ** 2)) + 0.005  # 0.005 is the tolarance
                heights[j] = max(heights[j], height)

    heights = np.clip(heights, height * 1, height * 2)

    # generate tooth according to calculated width and height
    tooth_func = [teeth_involute_sin(get_value_on_tooth_domain(i, tooth_samples),
                                     heights[get_teeth_idx(i, tooth_samples)], width=0.5) for i in
                  range(n)]

    deviations = np.array([[normals[i][0] * tooth_func[i], normals[i][1] * tooth_func[i]] for i in range(n)])
    return contour + deviations
