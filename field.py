import numba as nb
import numpy as np
from numba_kdtree import KDTree
import matplotlib.pyplot as plt


@nb.njit(parallel=True)
def _inverse_dist_weighting(indices, new, old, z, p):
    znew = np.empty(new.shape[0], dtype=new.dtype)
    for i in nb.prange(new.shape[0]):
        num = 0.0
        den = 0.0
        for j in indices[i]:
            d = np.sqrt((new[i, 0] - old[j, 0]) ** 2 +
                        (new[i, 1] - old[j, 1]) ** 2)
            if d > 0.0:
                num = num + np.power(d, - p) * z[j]
                den = den + np.power(d, - p)
            else:
                num = z[j]
                den = 1.0
                break
        # maybe there's a better way to handle this case
        znew[i] = num / den if (num > 0 and den > 0) else 0

    return znew


@nb.njit(parallel=True)
def _bilinear_weighting(indices, new, old, z):    
    znew = np.empty(new.shape[0], dtype=new.dtype)
    # for every point in the finer grid
    for i in nb.prange(new.shape[0]):
        x, y = new[i] # get coordinates of said point
        # initialize rectangel around said point
        corners = np.empty((4, 2)) 
        v_corners = np.empty((4, 2), dtype=z.dtype)
        # for each of the four closest corners
        for u, j in enumerate(indices[i]):
            corners[u] = old[j] # get their coordinates
            v_corners[u] = z[j] # get the value of the function at corners
        
        # I only need x, y coordinates
        x1, x2 = np.min(corners[:, 0]), np.max(corners[:, 0])
        y1, y2 = np.min(corners[:, 1]), np.max(corners[:, 1])
        
        # see wikipedia
        vec = np.array([1, x, y, x*y])
        mat = np.array([[x2*y2, -y2, -x2, 1.0], 
                       [-x2*y1, y1,  x2, -1.0], 
                       [-x1*y2, y2,  x1, -1.0], 
                       [x1*y1, -y1, -x1,  1.0]])
        
        # set the value at the point of the finer grid
        znew[i] = v_corners @ mat @ vec / (x2 - x1) / (y2 - y1)
        
    return znew

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

    def interpolate(self, xnew, ynew, p=2,
                    kdtree_kwargs={"leafsize": 10},
                    query_kwargs={"r": 0.08, "return_sorted": True}):

        self.normalize_radius()
        x, y = self.to_cartesian()
        old = np.c_[x, y]
        z = self.values.flatten()
        new = np.c_[xnew, ynew]
        
        kdtree = KDTree(old, **kdtree_kwargs)
        
        # if the radius of search, we look at just the 4 closest neighbours
        # p does not matter at that point, query_kwargs["k"] is set to 4
        do_bilinear = False
        if "r" in query_kwargs.keys():
            if query_kwargs["r"] == 0.0:
                do_bilinear = True
                query_kwargs.pop("r")
            else:
                do_bilinear = False
        else:
            do_bilinear = True
        
        if do_bilinear:
            query_kwargs["k"] = 4
            _, inds = kdtree.query(new, **query_kwargs)
            result = _bilinear_weighting(inds, new, old, z)
        else:
            inds = kdtree.query_radius(new, **query_kwargs)
            result = _inverse_dist_weighting(inds, new, old, z, p)
        
        return result
        
    def bil_interpolate(self, xnew, ynew, p=2,
                    kdtree_kwargs={"leafsize": 10},
                    query_kwargs={"k":4}):

        self.normalize_radius()
        x, y = self.to_cartesian()
        old = np.c_[x, y]
        z = self.values.flatten()
        kdtree = KDTree(old, **kdtree_kwargs)
        new = np.c_[xnew, ynew]
        _, inds = kdtree.query(new, **query_kwargs)

        return _bilinear_weighting(inds, new, old, z, p)

        

    def plot(self):
        fig = plt.figure()
        ax = fig.gca()
        x, y = self.to_cartesian()
        z = self.values.flatten()
        ax.tricontourf(x, y, z, levels=256)
        return fig, ax
