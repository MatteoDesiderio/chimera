"""
Several misc utilities
"""
import numpy as np
import numba as nb

def rms(x, axis=0):
    return np.sqrt(np.ma.sum(x**2, axis=axis) / len(x))

@nb.njit(parallel=True)
def to_polar(x, y):
    theta = np.empty(x.shape, dtype=x.dtype)
    r = np.empty(x.shape, dtype=x.dtype)
    for i in nb.prange(len(x)):
        r[i] = np.hypot(x[i], y[i])
        theta[i] = np.arctan2(y[i], x[i])
    return r, theta

def set_renormalized_fields(list_of_fields):
    sum_ = 0.0
    
    # TODO: fix case where sum_ = 0 
    # (all molten: Bs=Hz=Pr=0)
    for f in list_of_fields:
        sum_ += f.values
    for f in list_of_fields:
        f.values /= sum_
    
def to_cartesian(r, theta):
    """


    Returns
    -------
    TYPE tuple of two numpy.array.
        (x, y). The coordinates are unraveled.
        len(x) = len(y) = len(r) * len(theta) = self.values.size
    """
    r_grid, theta_grid = np.meshgrid(r, theta)
    z = r_grid * np.exp(1j * theta_grid)
    x, y = np.real(z).flatten(), np.imag(z).flatten()
    return x, y