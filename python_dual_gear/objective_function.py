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
