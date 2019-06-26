from subprocess import Popen, PIPE
import struct


def endian_test():
    try:
        with Popen(['endian_test'], stdin=PIPE, stdout=PIPE) as proc:
            out, err = proc.communicate(struct.pack('dic', 233.33, 1024, b'a'))
            return out == b'233.33000 01024 a'
    except Exception:
        return False  # any error


assert endian_test()  # shall be asserted whenever the package is imported


def polygon_to_bytes(polygon: [(float, float)]) -> bytes:
    """
    convert a polygon into bytes
    :param polygon: iterable points [(x,y)]
    :return: bytes with format: int32(size) double(x) double(y) double(x) double(y)
    """
    result = struct.pack('i', len(polygon))
    for point in polygon:
        x, y = point
        result = result + struct.pack('dd', x, y)
    return result


if __name__ == '__main__':
    target_polygon = [(1, 2), (3, 4), (5, 6)]
    gear_polygon = [(2.1, 2.7), (3.5, 3.4), (233.33, 233.33), (0.0, -1.0)]
    with Popen(['turning_function'], stdin=PIPE, stdout=PIPE) as proc:
        out, err = proc.communicate(polygon_to_bytes(target_polygon) + polygon_to_bytes(gear_polygon), 5)
        print(out.decode())
