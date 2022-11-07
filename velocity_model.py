import numpy as np
from numba import prange, njit
import pickle
from scipy.spatial import KDTree
from field import Field
from interfaces.axi.inparam_hetero_template import inparam_hetero

def voigt(moduli, compositions):
    sum_ = np.zeros(compositions[0].shape)
    for m, f in zip(moduli, compositions):
        sum_ += m * f
    return sum_


def reuss(moduli, compositions):
    sum_ = np.zeros(compositions[0].shape)
    for m, f in zip(moduli, compositions):
        sum_ += f / m
    return 1.0 / sum_


def compute_p(rho, K, G):
    return np.sqrt((K + 4 * G / 3) * 1e5 / rho)   # m / s


def compute_s(rho, G):
    return np.sqrt(G * 1e5 / rho)


def compute_bulk(rho, K):
    return np.sqrt(K * 1e5 / rho)                    # m / s


@njit(parallel=True)
def to_polar(x, y):
    theta = np.empty(x.shape, dtype=x.dtype)
    r = np.empty(x.shape, dtype=x.dtype)
    for i in prange(len(x)):
        r[i] = np.hypot(x[i], y[i])
        theta[i] = np.arctan2(y[i], x[i])
    return r, theta


class VelocityModel:
    def __init__(self, model_name, i_t, t, x, y, Cnames=list()):
        # model name, time info, radius of earth
        self.model_name = model_name
        self.i_t = i_t
        self.t = t
        self.r_E_km = 6371.0  # TODO initialize this from proj/field/other info
        # spatial coordinates
        self.x = x
        self.y = y
        r, th = to_polar(self.x, self.y)
        self.r, self.theta = r, th + np.pi / 2
        # compositional fields and corresponding names
        self.Cnames = Cnames
        self.C = []
        # T, P fields
        self.T = []
        self.P = []
        # velocity fields
        self.s = None
        self.p = None
        self.bulk = None
        # velocity anomaly fields
        self.s_a = None
        self.p_a = None
        self.bulk_a = None
        # average radial profiles
        self.s_prof = {"r": None, "val": []}
        self.p_prof = {"r": None, "val": []}
        self.bulk_prof = {"r": None, "val": []}
        # average fields
        self.K = None
        self.G = None
        self.rho = None
        self.rho_stagyy = None
        self.template = inparam_hetero # inparam_hetero template

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        self.C = np.empty((len(self.Cnames), len(value)))

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, value):
        self._P = value

    def compute_velocities(self, use_stagyy_rho):
        if use_stagyy_rho:
            rho = self.rho_stagyy
        else:
            rho = self.rho
        self.stagyy_rho_used = use_stagyy_rho

        K, G = self.K, self.G
        self.s = compute_s(rho, G)
        self.p = compute_p(rho, K, G)
        self.bulk = compute_bulk(rho, K)

    def load_moduli(self, path_moduli, proj_dict):
        name = self.model_name
        shape = self.C.shape
        K_list = np.empty(shape)
        G_list = np.empty(shape)
        rho_list = np.empty(shape)
        
        # TODO transfer the proj_dict from proj class to thermo_data class
        for i, nm in enumerate(self.Cnames):
            comp = proj_dict[nm]
            # print(nm, comp) # to check correct order of loading
            G_path = path_moduli + comp + "_" + "G" + ".npy"
            K_path = path_moduli + comp + "_" + "K" + ".npy"
            # TODO check if it's best to load the  from stagyy
            rho_path = path_moduli + comp + "_" + "rho" + ".npy"

            K_list[i] = np.load(K_path)
            G_list[i] = np.load(G_path)
            rho_list[i] = np.load(rho_path)

        return K_list, G_list, rho_list

    def average(self, K_list, G_list, rho_list):
        self.K = (reuss(K_list, self.C) + voigt(K_list, self.C)) / 2
        self.G = (reuss(G_list, self.C) + voigt(G_list, self.C)) / 2
        self.rho = voigt(rho_list, self.C)

    def vel_rho_to_npy(self, destination):
        model_name = self.model_name

        print('Saving seismic velocity fields in ' + destination)
        fname_s = model_name + "_Vs.npy"
        fname_p = model_name + "_Vp.npy"
        fname_b = model_name + "_Vb.npy"
        fname_rho = model_name + "_rho.npy"

        print(fname_s + " for shear wave velocity")
        np.save(destination + fname_s, self.s)
        print(fname_p + " for body wave velocity")
        np.save(destination + fname_p, self.p)
        print(fname_b + " for bulk sound velocity")
        np.save(destination + fname_b, self.bulk)
        print(fname_rho + " for average density")
        np.save(destination + fname_rho, self.rho)

    def get_rprofile(self, var="s", round_param=3):
        """
        

        Parameters
        ----------
        var : TYPE, optional
            DESCRIPTION. The default is "s".
        round_param : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        rsel : TYPE
            DESCRIPTION.
        prof : TYPE
            DESCRIPTION.

        """
        vel = getattr(self, var)
        # hacky way to deal with a serious problem, numerical precision
        rsel = np.sort(list(set(np.around(self.r, round_param))))

        diffs = np.diff(rsel)
        drmin = diffs[diffs > 0].min() / 2

        prof = np.empty(len(rsel))
        for i, r_i in enumerate(rsel):
            r1, r2 = r_i - drmin, r_i + drmin
            level = (self.r > r1) & (self.r < r2)
            prof[i] = np.mean(vel[level])

        return rsel, prof

    def anomaly(self, var="s", round_param=3):
        vel = getattr(self, var)
        rprof, vprof = self.get_rprofile(var, round_param)
        diffs = np.diff(rprof)
        drmin = diffs[diffs > 0].min() / 2

        tree = KDTree(np.c_[rprof, np.zeros(len(rprof))])
        other_tree = KDTree(np.c_[self.r, np.zeros(len(self.r))])
        indices = tree.query_ball_tree(other_tree, r=drmin)

        arr = np.empty(len(self.r))
        for i, index in enumerate(indices):
            for j in index:
                arr[j] = (vel[j] - vprof[i]) / vprof[i]

        setattr(self, var+"_a", arr)
        setattr(self, var+"_prof", {"r": rprof, "val": vprof})

        return getattr(self, var+"_a"), getattr(self, var+"_prof")

    @staticmethod
    def load(vmodel_path):
        with open(vmodel_path + 'v_model_data.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class

    def save(self, destination):
        with open(destination + 'v_model_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def export(self, destination, fmt):
        r, th = self.r * 1e3 * self.r_E_km, self.theta * 180 / np.pi
        th -= 180.0 # TODO check if it's always the same shift
        print("min/max thetas: %.1f, %.1f " % (th.min(), th.max()))
        if self.stagyy_rho_used:
            rho = self.rho_stagyy
        else:
            rho = self.rho
        s, p = self.s, self.p
        data = np.c_[r, th, p, s, rho]
        # save the sph text file
        fname = destination + "/geodynamic_hetfile.sph"
        np.savetxt(fname, data, header=str(len(data)), comments='', fmt=fmt)
        # save a corresponding inparam_hetero, as needed by axisem
        filled_template = self.template.format("geodynamic_hetfile.sph")
        with open(destination + "/inparam_hetero", "w") as inparam_file:
            inparam_file.write(filled_template)
        
