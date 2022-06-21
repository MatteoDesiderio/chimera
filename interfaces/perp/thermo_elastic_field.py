import numpy as np


class ThermoElasticField:
    def __init__(self, tab=None, label=None):
        self.tab = tab
        self.label = label if not (label is None) else tab.tab["title"]

        (T, P, rho_, K_, G_, _, _), _ = tab.to_ndarray()
        P *= 1e5                # since stagyy is in Pa (convert P [bar]->[Pa])
        rho = tab.remove_nans(rho_)
        K = tab.remove_nans(K_)
        G = tab.remove_nans(G_)
        self.T = T
        self.P = P
        self.rho = rho
        self.K = K
        self.G = G

    def extract(self, T_grid, P_grid, model_name):
        n_points = len(T_grid)
        # initialize empty arrays
        K_field = np.zeros(n_points)  # these'll be filled based on output
        G_field = np.zeros(n_points)  # from perplex
        rho_field = np.zeros(n_points)

        print("Retrieving moduli, density as function of P, T")
        # fill arrays: check in every cell of your models
        for i in range(n_points):

            # what are T, P conditions in each cell?
            T_i = T_grid[i]
            P_i = P_grid[i]
            # where are the closest T, P in the table?
            u = np.argmin(np.abs(T_i - self.T))
            v = np.argmin(np.abs(P_i - self.P))
            # select the corresponding property: f(T, P)
            K_field[i] = self.K[u, v]
            G_field[i] = self.G[u, v]
            rho_field[i] = self.rho[u, v]

        # save arrays
        fname = './rho_K_G/' + model_name + '_' + self.tab.tab["title"] + '_'
        print("Saving as %sXYZ_t%.1f.npy" % (fname, t))
        np.save(fname + ('rho_t%.1f' % t), rho_field)
        np.save(fname + ('Ks_t%.1f' % t), K_field)
        np.save(fname + ('Gs_t%.1f' % t), G_field)
        print("Done for rho, Ks, Gs")
#
