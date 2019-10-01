
from fabrication import read_2d_obj

def phisical_size(points:list):
    pt_x = []
    pt_y = []
    for point in points:
        pt_x.append(point[0])
        pt_y.append(point[1])
    max_x = max(pt_x)
    min_x = min(pt_x)
    max_y = max(pt_y)
    min_y = min(pt_y)
    return (max_x-min_x, max_y-min_y)

if __name__ == '__main__':
    filename_drive = './debug/drive_2d.obj'
    filename_driven = './debug/driven_2d.obj'
    drive_pt = read_2d_obj(filename_drive)
    driven_pt = read_2d_obj(filename_driven)
    drive_size = phisical_size(drive_pt)
    driven_size = phisical_size(driven_pt)
    distance = 0.5  # this should be provided and manually typed
    ratio = 8*7.97/distance # the real distance should also be manually typed
    print(drive_size[0]*ratio+'\n')
    print(drive_size[1]*ratio+'\n')
    print(driven_size[0]*ratio+'\n')
    print(driven_size[1]*ratio+'\n')
