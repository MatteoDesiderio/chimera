"""
Several misc utilities.
"""
import numba as nb
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import bartlett, triang, tukey
from scipy.stats import multivariate_normal


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

class Downsampler:
    def __init__(self, x, y, xnew, ynew):
        self.x, self.y = x, y
        self.xnew = xnew
        if ynew.min() < 0:
            self.ynew = np.unwrap(ynew)
        else:
            self.ynew = ynew

    def downsample(self, z, method="avg"):

        if method=="gauss":
            # 1) resample original mesh (dr is not constant)
            dx = np.diff(np.abs(self.x)).min()
            dy = np.diff(np.abs(self.y)).min()
            shape = self.y.size, self.x.size
            f = interp1d(self.x, z.reshape(shape))
            x_int = np.arange(self.x.min(), self.x.max() + dx, dx)
            x_int[-1] = 1
            z_int = f(x_int)
            ny, nx = z_int.shape

            xx, yy = np.meshgrid(x_int, self.y)
            pos = np.dstack((xx, yy))

            znew = np.zeros([self.xnew.size, self.ynew.size])
            xdiffs = np.abs(np.diff(self.xnew))
            xdiffs = np.r_[xdiffs, xdiffs[-1]] / 2
            ydiffs = np.abs(np.diff(self.ynew))
            ydiffs = np.r_[ydiffs, ydiffs[-1]] / 2

            for ix, mux in enumerate(self.xnew):
                varx = xdiffs[ix] ** 2
                for iy, muy in enumerate(self.ynew):
                    vary = ydiffs[iy] ** 2
                    mu = [mux, muy]
                    var = [[varx, 0], [0, vary]]
                    weights = multivariate_normal(mu, var).pdf(pos)
                    weights /= weights.max()
                    znew[ix, iy] = np.sum(weights * z_int) / np.sum(weights)

            return None, znew.T


        elif method=="fourier":
            # 1) resample original mesh (dr is not constant)
            dx = np.diff(np.abs(self.x)).min()
            dy = np.diff(np.abs(self.y)).min()
            shape = self.y.size, self.x.size
            f = interp1d(self.x, z.reshape(shape))
            x_int = np.arange(self.x.min(), self.x.max() + dx, dx)
            x_int[-1] = 1
            z_int = f(x_int)
            ny, nx = z_int.shape

            # 2) fourier transform
            tr_z_int = np.fft.fft2(z_int)
            phase = np.angle(tr_z_int)
            kx = np.fft.fftfreq(nx, dx)
            ky = np.fft.fftfreq(ny, dy)

            # 3) antialias filter
            ky_nyq =  1 / np.diff(self.ynew).max() / 2
            kx_nyq =  1 / np.diff(self.xnew).max() / 2

            rectangle = np.zeros(tr_z_int.shape)

            y_range = np.abs(ky) < ky_nyq
            x_range = np.abs(kx) >= kx_nyq
            rectangle[y_range] = 1
            rectangle[:, x_range] = 0
            alpha = .5
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
            filt_tr_z_int = filt * np.abs(tr_z_int) * np.exp(1j*phase)
            filt_z_int = np.real(np.fft.ifft2(filt_tr_z_int))

            xx, yy = to_cartesian(x_int, self.y)
            coords = np.c_[xx, yy]

            return coords, filt_z_int

        else:
            # 1) resample original mesh (dr is not constant)
            dx = np.abs(np.diff(self.x)).min()
            dy = np.abs(np.diff(self.y)).min()
            shape = self.y.size, self.x.size
            f = interp1d(self.x, z.reshape(shape))
            x_int = np.arange(self.x.min(), self.x.max() + dx, dx)
            x_int[-1] = 1
            z_int = f(x_int)
            ny, nx = z_int.shape

            xx, yy = np.meshgrid(x_int, self.y)
            pos = np.dstack((xx, yy))

            znew = np.zeros([self.xnew.size, self.ynew.size])
            xdiffs = np.abs(np.diff(self.xnew))
            xdiffs = np.r_[xdiffs, xdiffs[-1]] / 2
            ydiffs = np.abs(np.diff(self.ynew))
            ydiffs = np.r_[ydiffs, ydiffs[-1]] / 2

            for ix, mux in enumerate(self.xnew):
                sx = xdiffs[ix]
                for iy, muy in enumerate(self.ynew):
                    sy = ydiffs[iy]
                    weights = np.zeros_like(z_int)
                    squarex = (xx >= mux - sx) & (xx <= mux + sx)
                    squarey = (yy >= muy - sy) & (yy <= muy + sy)
                    square = squarex & squarey
                    weights[square] = 1.0
                    if method == "triang":
                        mx = len(squarex[0, :][squarex[0, :]])
                        my = len(squarey[:, 0][squarey[:, 0]])
                        triangx = bartlett(mx)
                        triangy = triang(my)
                        pyramid = triangy[:, np.newaxis] * triangx
                        weights[square] *= pyramid.flatten()
                    znew[ix, iy] = np.sum(weights * z_int) / np.sum(weights)
            return None, znew.T
