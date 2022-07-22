import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class Field:
    """
    General class implementing a named field defined on a regular grid. 
    The coordinates of the grid must be polar (r, theta).
    """

    def __init__(self, name="T"):
        self._coords = (None, None)
        self._values = None
        self.name = name
        self.r_max = None

    @property
    def coords(self):
        return self._coords

    @property
    def values(self):
        return self._values

    @property
    def polar(self):
        return self._polar

    @polar.setter
    def polar(self, value):
        self._polar = value

    @coords.setter
    def coords(self, value):
        if len(value) != 2:
            raise TypeError("Coords must be a 2-elements tuple.")
        self._coords = value

    @values.setter
    def values(self, value):
        self._values = value

    def to_cartesian(self):
        """


        Returns
        -------
        TYPE tuple of two numpy.array.
            (x, y). The coordinates are unraveled.
            len(x) = len(y) = len(r) * len(theta) = self.values.size
        """
        r_grid, theta_grid = np.meshgrid(*self.coords)
        z = r_grid * np.exp(1j * theta_grid)
        x, y = np.real(z).flatten(), np.imag(z).flatten()
        return x, y

    def split(self, edge_pad_perc=0.25):
        """
        Splits the model in two portions (a left one and right one). 
        This is needed, as the axisem grid is only half of the whole "annulus".
        Using edge_extension_perc it is possible to pad and include more points 
        from boht sides of the halved field. This is useful when interpolating 
        on the final fine grid.

        Parameters
        ----------
        edge_pad_perc : float
            Extent of the padding, given as a percentage of the half-length of 
            self.values. The number of indices to get from each side of the 
            half-annulus is int(edge_pad_perc * self.values.shape[0] // 2).

        Returns
        -------
        fields : list of 2 objects of type(self)
            The two portions of the stagyy field: [left, right].

        """
        # I divide in half
        nth_2 = self.values.shape[0] // 2
        # then i take a portion of that half defined by my percentage
        dx_2 = int(edge_pad_perc * nth_2)
        # the stuff that's on top appears at the bottom and viceversa
        left_half = np.roll(self.values, -dx_2, axis=0)[nth_2 - 2 * dx_2:]
        right_half = np.roll(self.values, dx_2, axis=0)[:nth_2 + 2 * dx_2]
        r, theta = self.coords
        # I need to decide whether I want to tak that from the existing
        # attributes or if I want to recreate them
        theta_r = np.roll(theta, -dx_2)[nth_2 - 2 * dx_2:]
        theta_l = np.roll(theta, dx_2)[:nth_2 + 2 * dx_2]
        fields = [Field(), Field()]
        for fld, half, coord in zip(fields,
                                    [left_half, right_half],
                                    [(r, theta_l), (r, theta_r)]):
            fld.values = half
            fld.coords = coord

        return fields

    def normalize_radius(self):
        self.r_max = self.coords[0].max()
        self._coords = self.coords[0] / self.r_max, self.coords[1]

    def interpolate(self, interp_type, xnew, ynew):

        self.normalize_radius()
        x, y = self.to_cartesian()
        old = np.c_[x, y]
        z = self.values.flatten()
        new = np.c_[xnew, ynew]
        interpolated = griddata(old, z, new, method=interp_type)
        
        return interpolated
        

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        x, y = self.to_cartesian()
        z = self.values.flatten()
        ax.tricontourf(x, y, z, levels=256)
        return fig, ax
