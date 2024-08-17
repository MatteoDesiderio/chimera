import numpy as np
from numba import njit, prange
from numba_kdtree import KDTree
from scipy.interpolate import griddata


def _TP_to_xy(T_tab, P_tab):
    lenT, lenP = len(T_tab), len(P_tab)
    x, y = np.tile(T_tab, lenP), np.tile(P_tab, lenT)
    x, y = np.meshgrid(T_tab, P_tab)
    return x.flatten(), y.flatten()

def _gridder(x, y, T_axi, P_axi, field_tab, kwargs):
    return griddata(np.c_[x, y], field_tab.flatten(), (T_axi, P_axi), **kwargs)

@njit(parallel=True)
def _store(inds, fld):
    lngth = len(inds)
    arr = np.zeros(lngth, dtype=fld.dtype)
    for i in prange(lngth):
        arr[i] = fld[inds[i][0]]
    return arr

class ThermoElasticField:
    def __init__(self, tab=None, label=None):
        self.tab = tab
        self.label = label if label is not None else tab.tab["title"]
        self.rho = None
        self.K = None
        self.G = None

    def extract(self, inds, model_name):
        """


        Parameters
        ----------
        inds : TYPE
            DESCRIPTION.
        model_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        rho, K, G = self.tab.data[2:]

        self.rho = _store(inds, rho.T.flatten())
        self.K = _store(inds, K.T.flatten())
        self.G = _store(inds, G.T.flatten())

    def save(self, path):
        fname = path + self.tab.tab["title"] + "_"
        print(f"Saving as {fname}<parameter>.npy")
        np.save(fname + "rho",  self.rho)
        np.save(fname + "K", self.K)
        np.save(fname + "G", self.G)
        print("Done for rho [kg/m^3], Ks [bar], Gs [bar]")

    @staticmethod
    def get_tree(tab):
        """


        Parameters
        ----------
        tab : TYPE
            DESCRIPTION.

        Returns
        -------
        kdtree : TYPE
            DESCRIPTION.

        """
        print("Creating a tree out of the P, T range of the thermodynamic",
              "dataset")
        T, P = tab.data[:2]
        # since stagyy is in Pa, convert P [bar]->[Pa])
        P *= 1e5
        x, y = _TP_to_xy(T, P)
        kdtree = KDTree(np.c_[x, y], leafsize=10)
        print("KDTree Created")
        return kdtree

    @staticmethod
    def get_indices(tree, T_grid, P_grid):
        """


        Parameters
        ----------
        tree : TYPE
            DESCRIPTION.
        T_grid : TYPE
            DESCRIPTION.
        P_grid : TYPE
            DESCRIPTION.

        Returns
        -------
        inds : TYPE
            DESCRIPTION.

        """
        print("Retrieving moduli, density as function of P, T")
        # _, inds = tree.query(np.c_[T_grid, P_grid]) # changed with v0.1.6 # noqa: ERA001
        # number of valid neighbors is returned too now. Must be ignored
        _, inds, _ = tree.query(np.c_[T_grid, P_grid])
        print("KDTree queried")
        return inds
