import numpy as np
from numba import prange, njit
import pickle

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
        self.model_name = model_name
        self.i_t = i_t
        self.t = t
        # spatial coordinates
        self.x = x
        self.y = y
        self.r, self.theta = to_polar(self.x, self.y)
        self.theta += np.pi / 2
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
        # average fields
        self.K = None
        self.G = None
        self.rho = None

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
   
    def compute_velocities(self):
        rho, K, G = self.rho, self.K, self.G
        self.s = compute_s(rho, G)
        self.p = compute_p(rho, K, G)
        self.bulk = compute_bulk(rho, K)  
    
    def load_moduli(self, path_moduli, proj_dict):
        name = self.model_name
        shape = self.C.shape
        K_list = np.empty(shape)
        G_list = np.empty(shape)
        rho_list = np.empty(shape)
        
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
        print("FIX (VOIGT + REUSS) / 2 != REUSS + VOIGT / 2")
        self.K = reuss(K_list, self.C) + voigt(K_list, self.C) / 2
        self.G = reuss(G_list, self.C) + voigt(G_list, self.C) / 2
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
    
    @staticmethod
    def load(project_path):
        with open(project_path + 'v_model_data.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class

    def save(self, destination):
        with open(destination + 'v_model_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        