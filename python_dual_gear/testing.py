"""
Functions used for testing
"""
from typing import List, Tuple, Union
import numpy as np
from debug_util import MyDebugger, SubprocessDebugger
from opt_dual_shapes import sampling_optimization


def _optimize_pair(drive_contour, driven_contour, debugger):
    sampling_optimization(drive_contour, driven_contour, 1, (4, 4), 1, 1024, 64, debugger,
                          max_iteration=5, max_sample_depth=3,
                          visualization={}, draw_tar_functions=True)


def optimization_test(names: List[List[str]], optimize_pairs: List[Tuple[np.ndarray, np.ndarray]],
                      parallel: bool = False):
    """
    test optimization for multiple pairs in sub-processes
    :param names: names for each pair for debug use
    :param optimize_pairs: the pairs to be optimized to
    :param parallel: run subprocesses in parallel
    :return:
    """
    processes = []
    for name, pair in zip(names, optimize_pairs):
        debugger = MyDebugger(name)
        process = SubprocessDebugger(debugger, _optimize_pair, (*pair, debugger))
        process.start()
        if parallel:
            processes.append(process)
        else:
            process.join()
    for process in processes:
        process.join()


if __name__ == '__main__':
    import math

    square_contour = np.array(
        [(0, 0), (10, 0), (10, 10), (0, 10)]
    )
    circle_contour = np.array(
        [(5 * math.cos(theta), 5 * math.sin(theta)) for theta in np.linspace(0, 2 * math.pi, 1024, endpoint=False)])
    optimization_test([['circle', 'circle'], ['circle', 'square'], ['square', 'square']],
                      [(circle_contour, circle_contour), (circle_contour, square_contour),
                       (square_contour, square_contour)], True)
