import numpy as np
#from numba_kdtree import KDTree
from scipy.spatial import KDTree
import numba as nb


class ThermoElasticField:
    def __init__(self, tab=None, label=None):
        self.tab = tab
        self.label = label if not (label is None) else tab.tab["title"]
        self.rho = None
        self.K = None
        self.G = None

    def extract(self, T_grid, P_grid, model_name): #, 
        T, P, rho, K, G = self.tab.data
        n_points = len(T_grid)
        # since stagyy is in Pa, convert P [bar]->[Pa])
        P *= 1e5
        # initialize empty arrays
        n_points = len(T_grid)
        K_field = np.empty(n_points, dtype=T.dtype)  # these'll be filled 
        G_field = np.empty(n_points, dtype=T.dtype)  # from perplex
        rho_field = np.empty(n_points, dtype=T.dtype)

        print("Retrieving moduli, density as function of P, T")
        # fill arrays: check in every cell of your models
        
        for i in range(n_points):
            # what are T, P conditions in each cell?
            T_i = T_grid[i]
            P_i = P_grid[i]
            # where are the closest T, P in the table?
            u = np.argmin(np.abs(T_i - T))
            # since stagyy is in Pa, convert P [bar]->[Pa])
            v = np.argmin(np.abs(P_i - P))
            # select the corresponding property: f(T, P)
            K_field[i] = K[u, v]
            G_field[i] = G[u, v]
            rho_field[i] = rho[u, v]
            
        self.rho = rho_field
        self.G = G_field
        self.K = K_field
    
    def extractKD(self, T_grid, P_grid, model_name):  
        pass
    
    def save(self, path):
        fname = path + self.tab.tab["title"] + '_'
        print("Saving as %s<parameter>.npy" % fname)
        np.save(fname + 'rho',  self.rho)
        np.save(fname + 'K', self.K)
        np.save(fname + 'G', self.G)
        print("Done for rho [kg/m^3], Ks [bar], Gs [bar]")
