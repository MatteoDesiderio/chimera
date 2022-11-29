import numpy as np
from scipy.interpolate import griddata
from numba_kdtree import KDTree
from numba import prange, njit

def _TP_to_xy(T_tab, P_tab):
    x, y = np.meshgrid(T_tab, P_tab)
    return x.flatten(), y.flatten()

def _gridder(x, y, T_axi, P_axi, field_tab, kwargs):
    return griddata(np.c_[x, y], field_tab.flatten(), (T_axi, P_axi), **kwargs)

@njit(parallel=True)
def order():
    for value in prange(5):
        print(value)

@njit(parallel=True)
def store(inds, fld):
    lngth = len(inds)
    arr = np.zeros(lngth, dtype=fld.dtype)
    for i in prange(lngth):
        arr[i] = fld[inds[i][0]]
    return arr


class ThermoElasticField:
    def __init__(self, tab=None, label=None):
        self.tab = tab
        self.label = label if not (label is None) else tab.tab["title"]
        self.rho = None
        self.K = None
        self.G = None
        
    @staticmethod
    def get_tree(tab):
        T, P = tab.data[:2]
        # since stagyy is in Pa, convert P [bar]->[Pa])
        P *= 1e5  
        x, y = _TP_to_xy(T, P)
        kdtree = KDTree(np.c_[x, y], leafsize=10)
        print("KDTree Created", end=" ")
        return kdtree
        
    def extract(self, T_grid, P_grid, model_name): #, 
        T, P, rho, K, G = self.tab.data
        # since stagyy is in Pa, convert P [bar]->[Pa])
        P *= 1e5

        print("Retrieving moduli, density as function of P, T")
        x, y = _TP_to_xy(T, P)
        kdtree = KDTree(np.c_[x, y], leafsize=10)
        _, ii = kdtree.query(np.c_[T_grid, P_grid])
        print("KDTree queried")
        
        self.rho = store(ii, rho.flatten())
        self.K =  store(ii, K.flatten())
        self.G = store(ii, G.flatten())
    
    def save(self, path):
        fname = path + self.tab.tab["title"] + '_'
        print("Saving as %s<parameter>.npy" % fname)
        np.save(fname + 'rho',  self.rho)
        np.save(fname + 'K', self.K)
        np.save(fname + 'G', self.G)
        print("Done for rho [kg/m^3], Ks [bar], Gs [bar]")
