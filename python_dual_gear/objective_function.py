from subprocess import Popen, PIPE
import struct

if __name__ == '__main__':
    with Popen(['echo'], stdin=PIPE, stdout=PIPE) as proc:
        out, err = proc.communicate(struct.pack('dd', 233.33, 233.33))
        for data in struct.iter_unpack('dd', out):
            print(data)
        print(repr(out))
