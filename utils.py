"""
Several misc utilities
"""
import numpy as np
import numba as nb


@nb.njit(parallel=True)
def to_polar(x, y):
    theta = np.empty(x.shape, dtype=x.dtype)
    r = np.empty(x.shape, dtype=x.dtype)
    for i in nb.prange(len(x)):
        r[i] = np.hypot(x[i], y[i])
        theta[i] = np.arctan2(y[i], x[i])
    return r, theta

def set_renormalized_fields(f1, f2, f3):
    v1, v2, v3 = f1.values, f2.values, f3.values
    v2 = 1 - v1
    sum_ =  v1 + v2 + v3
    v1 /= sum_
    v2 /= sum_
    v3 /= sum_
    f1.values, f2.values, f3.values = v1, v2, v3
    
