"""
functions related to 3D modelling and fabrication
"""


def generate_2d_obj(filename, points):
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


if __name__ == '__main__':
    from plot_sampled_function import polar_to_rectangular
    from drive_gears.ellipse_gear import generate_gear

    generate_2d_obj('test.obj', polar_to_rectangular(generate_gear(8192), None))
