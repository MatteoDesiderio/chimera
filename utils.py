"""
Several misc utilities
"""
import numpy as np
import numba as nb
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from skimage import transform

@nb.njit(parallel=True)
def to_polar(x, y):
    theta = np.empty(x.shape, dtype=x.dtype)
    r = np.empty(x.shape, dtype=x.dtype)
    for i in nb.prange(len(x)):
        r[i] = np.hypot(x[i], y[i])
        theta[i] = np.arctan2(y[i], x[i])
    return r, theta

def set_renormalized_fields(list_of_fields, bshzpr=True):
    sum_ = 0.0
    names = [f.name for f in list_of_fields]
    
    if bshzpr:
        if 'bs' in names and 'hz' in names and 'prim' in names:
            bs = list(filter(lambda x: x.name == "bs", list_of_fields))[0]
            pr = list(filter(lambda x: x.name == "prim", list_of_fields))[0]
            for f in list_of_fields:
                if f.name == 'hz':
                    f.values = 1 - bs.values
                    
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

class Downsampler:
    def __init__(self, x, y, xnew, ynew):        
        self.x, self.y = x, y
        self.xnew, self.ynew = xnew, ynew 
        
    def downsample(self, z):

        # 1) resample original mesh (dr is not constant)
        dx = np.diff(np.abs(self.x)).min()
        dy = np.diff(np.abs(self.y)).min()
        shape = self.y.size, self.x.size
        f = interp1d(self.x, z.reshape(shape))
        x_int = np.arange(self.x.min(), self.x.max() + dx, dx)
        x_int[-1] = 1
        z_int = f(x_int)
        ny, nx = z_int.shape
        
        # demean
        mean = np.mean(z_int)
        
        # 2) fourier transform
        tr_z_int = np.fft.fft2(z_int)
        kx = np.fft.fftfreq(nx, dx)
        ky = np.fft.fftfreq(ny, dy)
        
        # 3) antialias filter 
        ky_nyq =  1 / np.diff(self.ynew).max()
        kx_nyq =  1 / np.diff(self.xnew).max()
        
        rectangle = np.zeros(tr_z_int.shape)

        y_range = np.abs(ky) < ky_nyq
        x_range = np.abs(kx) >= kx_nyq
        rectangle[y_range] = 1
        rectangle[:, x_range] = 0
        alpha = .2
        x_win = tukey(np.count_nonzero( ~ x_range), alpha)
        half_nx = x_win.size // 2 + 1 * (x_win.size % 2 != 0)
        x_win = np.r_[x_win[half_nx-1:], 
                      np.ones(rectangle.shape[-1] - x_win.size) * x_win[[-1 ]], 
                      x_win[:half_nx-1]]
        y_win = tukey(np.count_nonzero(y_range), alpha)
        half_ny = y_win.size // 2 + 1 * (y_win.size % 2 != 0)
        y_win = np.r_[y_win[half_ny-1:], 
                      np.ones(rectangle.shape[0] - y_win.size) * y_win[[-1 ]], 
                      y_win[:half_ny-1]]
        filt = x_win * rectangle
        filt *= (y_win * filt.T).T
        filt[filt < 0] = 0
        
        # 4) apply filter and inv transform
        filt_tr_z_int = filt * np.abs(tr_z_int) * np.exp(1j*np.angle(tr_z_int)) 
        filt_z_int = np.real(np.fft.ifft2(filt_tr_z_int)) + mean
        
        xx, yy = to_cartesian(x_int, self.y)
        coords = np.c_[xx, yy]
        return coords, filt_z_int
        
    