import numpy as np

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

class VelocityModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.s = None
        self.p = None
        self.bulk = None
        # average fields
        self.K = None
        self.G = None
        self.rho = None
        
    def compute_velocities(self):
        rho, K, G = self.rho, self.K, self.G
        self.s = compute_s(rho, G)
        self.p = compute_p(rho, K, G)
        self.bulk = compute_bulk(rho, K)  
        
    def average(self, path_moduli, compositions, names, proj_dict):
        name = self.model_name
        shape = compositions.shape
        K_list = np.empty(shape)
        G_list = np.empty(shape)
        rho_list = np.empty(shape)
        
        for i, nm in enumerate(names):
            comp = proj_dict[nm]
            # print(nm, comp) # to check correct order of loading
            G_path = path_moduli + name + "_" + comp + "_" + "G" + ".npy"
            K_path = path_moduli + name + "_" + comp + "_" + "K" + ".npy"
            rho_path = path_moduli + name + "_" + comp + "_" + "rho" + ".npy"
            
            K_list[i] = np.load(K_path)
            G_list[i] = np.load(G_path)
            rho_list[i] = np.load(rho_path)
        
        self.K = reuss(K_list, compositions) + voigt(K_list, compositions) / 2
        self.G = reuss(G_list, compositions) + voigt(G_list, compositions) / 2
        self.rho = voigt(rho_list, compositions)
    
    def save(self, destination):
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
        