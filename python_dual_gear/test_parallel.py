import multiprocessing
import time
import math


try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 2   # arbitrary default


def square(n):
    return n * n **3 / (n+1)

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=cpus)
    start = time.time()
    lst = pool.map(square, list(range(10000)))
    end = time.time()
    print("MP time use:", end - start)

    start = time.time()
    lst = [square(x) for x in range(10000)]
    end = time.time()
    print("comprehension time use:", end - start)

    start = time.time()
    lst = list(map(square, list(range(10000))))
    end = time.time()
    print("mapping time use:", end - start)