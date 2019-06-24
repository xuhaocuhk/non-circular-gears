from scipy.optimize import dual_annealing
import numpy as np
import math


if __name__ == '__main__':
    # func = lambda x: np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
    func = lambda x: math.cos(x)
    lb = -100
    ub = 100
    ret = dual_annealing(func, bounds=[(lb,ub),], seed=3)
    print( f"global minimum: xmin = {ret.x}, f(xmin) = {ret.fun}" )






