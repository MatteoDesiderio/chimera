import numpy as np
from numba import prange, njit
import pickle
from scipy.spatial import KDTree
from field import Field
from interfaces.axi.inparam_hetero_template import inparam_hetero
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
import h5py
import os

# TO DO to speed things up first
# create a function that uses griddata to interpolate from the 1D prof to the
# whole grid. Thene use numba to compute the anomaly (loop over each single 
# point or divide the array into chunks and loop over each chunk in parallel)
@njit
def _anomaly(rprof, vprof, rmod, vmod, dr):
    arr = np.zeros_like(vmod)
    for iprof in prange(len(rprof)):
        r_pos = rprof[iprof]
        v_pos = vprof[iprof]
        imods = np.argwhere(np.abs(r_pos - rmod) <= dr)
        for imod in imods:
            arr[imod] = (vmod[imod] - v_pos) / v_pos
    return arr

def get_prof_pert(ext_z, ext_profs, z, profs, interp_kwargs={}):
    ext_profs_pert = []
    for ext_pr, pr in zip(ext_profs, profs):
        f = interp1d(ext_z, ext_pr, **interp_kwargs)
        ext_pr_i = f(z)
        pert = 100 * (pr - ext_pr_i) / pr
        ext_profs_pert.append(pert)
    return z, ext_profs_pert

def get_ext_prof(path, r_core_m=3481e3, r_Earth_m=6371e3, usecols=(0,3)):
    """
    Return vs, vp, density out of an external 1D model (an axisem .bm file) and 
    their depth coordinates in kms.

    Parameters
    ----------
    path : str
        Path of an external 1D model, an axisem .bm file 
        (caution: it is assumed that the first 6 lines are the header. 
         It is also assumed that the model is isotropic, r units are in
         meters and columns are r, rho, vpv, vsv, qka, qmu). 
    r_core_m : float, optional
        DESCRIPTION. The default is 3481e3.
    r_Earth_m : float, optional
        DESCRIPTION. The default is 6371e3.
    usecols : tuple, optional
        Use the columns from usecols[0] to usecols[-1]. The default is (0,3).
        

    Returns
    -------
    zprem_km  : numpy.ndarray
        Array containing the depth coordinates in kms.
    profs : list
        Values obtained from the 1D model supplied. 
        By default, vs, vp, density are returned.
        The parameter usecols may be used to trim/extend this list.
        The extended list is vs, vp, density, qka, qmu
        
        

    """
    rprem, rhoprem, vpprem, vsprem, qka, qmu = np.loadtxt(path, 
                                                          skiprows=6, 
                                                          unpack=True)
    mantle = rprem >= r_core_m
    rprem, rhoprem = rprem[mantle], rhoprem[mantle]
    vpprem, vsprem = vpprem[mantle], vsprem[mantle]
    qka, qmu = qka[mantle], qmu[mantle]

    zprem_km = (r_Earth_m - rprem) / 1e3
    profs = [vsprem, vpprem, rhoprem, qka, qmu]
    return zprem_km, profs[usecols[0]:usecols[1]]

def _is_quick_mode_on(_self):
    quick_mode_on = True
    try:
        quick_mode_on = _self.proj.quick_mode_on
    except AttributeError:
        quick_mode_on = False
        
    return quick_mode_on

def _create_labels(variables, absolute):
    def define_label(v):
        if v == "T":
            unit = "[K]" if absolute else "[%]" 
            return r"T " + unit
        if v == "s":
            unit = "[m/s]" if absolute else " (V - V1D)/V [%]" 
            return r"$V_s$ " + unit
        elif v == "p":
            unit = "[m/s]" if absolute else " (V - V1D)/V [%]"
            return r"$V_p$ " + unit
        else:
            unit = "[kg/m$^3$]" if absolute else " (rho - rho1D)/rho [%]"
            return r"$\rho$ "  + unit
    
    labels = []
    for v in variables:
        label = define_label(v)
        labels.append(label)
    return labels

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
    def __init__(self, model_name, i_t, t, x, y, Cnames=list(), proj=None):
        # model name, time info, radius of earth, other info
        self.model_name = model_name
        self.i_t = i_t
        self.t = t
        self.r_E_km = 6371.0  # TODO initialize this from proj/field/other info
        self.proj = proj
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
        # density / velocity anomaly fields
        self.rho_a = None
        self.rho_stagyy_a = None
        self.s_a = None
        self.p_a = None
        self.bulk_a = None
        # average radial profiles
        self.rho_prof = {"r": None, "val": []}
        self.rho_stagyy_prof = {"r": None, "val": []}
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

    def get_rprofile(self, var="s", round_param=3, shape=None):
        """
        

        Parameters
        ----------
        var : TYPE, optional
            DESCRIPTION. The default is "s".
        round_param : TYPE, optional
            DESCRIPTION. The default is 3.
        shape : None or list-like
            If list-like, will attempt to reshape the field into the provided
            shape before computation. Useful for custom grids 
            (i.e. neither axisem or stag)
        
        Returns
        -------
        rsel : TYPE
            DESCRIPTION.
        prof : TYPE
            DESCRIPTION.

        """
        # HACK to make it work with a previous version where the velocity
        # model did not have the attribute quick_mode_on
        quick_mode_on = _is_quick_mode_on(self)
            
        # if you have a regular grid, this operation is easier
        if quick_mode_on:
            rsel, vel = self.r, getattr(self, var)
            shape = [self.proj.geom["n{}tot".format(c)] for c in ("yz") ]
            shape[0] = shape[0] + 1
            rsel, vel = [ar.reshape(shape) for ar in (rsel, vel)]
            rsel = rsel[0]
            prof = np.mean(vel, axis=0)
            return rsel, prof
        
        else:
            if shape is None:
                # print(var.capitalize(), "profile for", self.model_name)
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
            else:
                rsel, vel = self.r, getattr(self, var)
                rsel, vel = [ar.reshape(shape) for ar in (rsel, vel)]
                rsel = rsel[0]
                prof = np.mean(vel, axis=0)
            return rsel, prof

    def anomaly(self, var="s", round_param=3, fac=100.0, shape=None):
        """
        

        Parameters
        ----------
        var : TYPE, optional
            DESCRIPTION. The default is "s".
        round_param : TYPE, optional
            DESCRIPTION. The default is 3.
        fac : TYPE, optional
            DESCRIPTION. The default is 100.0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        vel = getattr(self, var)
        rprof, vprof = self.get_rprofile(var, round_param, shape=shape)
        quick_mode_on = _is_quick_mode_on(self)
        
        if quick_mode_on:
            shape = [self.proj.geom["n{}tot".format(c)] for c in ("yz") ]
            shape[0] = shape[0] + 1
            vel = vel.reshape(shape)
            arr = fac * (vel - vprof) / vprof
            setattr(self, var+"_a", arr.flatten())
            setattr(self, var+"_prof", {"r": rprof, "val": vprof})
        else:
            if shape is None:
                diffs = np.diff(rprof)
                drmin = diffs[diffs > 0].min() / 2
                
                """
                tree = KDTree(np.c_[rprof, np.zeros(len(rprof))])
                other_tree = KDTree(np.c_[self.r, np.zeros(len(self.r))])
                indices = tree.query_ball_tree(other_tree, r=drmin)
                arr = np.empty(len(self.r))
                for i, index in enumerate(indices):
                    for j in index:
                        arr[j] = fac * (vel[j] - vprof[i]) / vprof[i]
                """
                
                arr = fac * _anomaly(rprof, vprof, self.r, vel, drmin)
    
                setattr(self, var+"_a", arr)
                setattr(self, var+"_prof", {"r": rprof, "val": vprof})
            else:
                vel = vel.reshape(shape)
                arr = fac * (vel - vprof) / vprof
                setattr(self, var+"_a", arr.flatten())
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

    def export(self, destination, fmt, absolute=True, fac=100, 
               fname="geodynamic_hetfile.sph", dtype="float32"):
        """
        

        Parameters
        ----------
        destination : TYPE
            DESCRIPTION.
        fmt : TYPE
            DESCRIPTION.
        absolute : TYPE, optional
            DESCRIPTION. The default is True.
        fac : TYPE, optional
            DESCRIPTION. The default is 100.
        fname : TYPE, optional
            DESCRIPTION. Accepted file formats are .sph and .hdf5. 
            The default is "geodynamic_hetfile.sph".
        dtype : TYPE, optional
            DESCRIPTION. Used only if file is hdf5. The default is "float".

        Returns
        -------
        None.

        """
        r, th = self.r * self.r_E_km, self.theta * 180 / np.pi
        th -= 90.0 # TODO check if it's always the same shift
        
        val_type = "" if absolute else "_a"
        adj = "absolute values" if absolute else "relative perturbations"
        print("Exporting model as %s" % adj)
        if not absolute:
            _ = [self.anomaly(var, fac=fac) for var in ["s", "p", "rho"]]
        
        print("min/max thetas: %.1f, %.1f " % (th.min(), th.max()))
        
        if self.stagyy_rho_used:
            rho = getattr(self, "rho_stagyy" + val_type)
        else:
            rho = getattr(self, "rho" + val_type)
            
        s, p = getattr(self, "s" + val_type), getattr(self, "p" + val_type)
        
        data = np.c_[r, th, p, s, rho]
        # save the sph text file
        fpath = destination + "/" + fname
        name, extension = fname.rsplit(".")
        if extension == "sph":
            np.savetxt(fpath,data, header=str(len(data)), comments='', fmt=fmt)
        else:
            if os.path.exists(fpath):
                raise OSError("hdf5 file with this name already exists.")
            else:
                for d, d_name in zip(data.T, ["r", "theta", "p", "s", "rho"]):
                    with h5py.File(fpath, "a") as hf:
                        hf.create_dataset(d_name, data=d, dtype=dtype)

        # save a corresponding inparam_hetero, as needed by axisem
        filled_template = self.template.format(fname)
        with open(destination + "/inparam_hetero", "w") as inparam_file:
            inparam_file.write(filled_template)
        
    def plot_profiles(self, variables=["s", "p", "rho"], fig=None, axs=None,
                      absolute=True, external=None,
                      interp_kwargs={"kind": "linear", "bounds_error": False,
                                     "fill_value": "extrapolate"}, 
                      third_variable=None,
                      plot_kwargs={"color": "r"}, 
                      cmap_kwargs={"cmap": "hot", "vmin": 300, "vmax": 4000}):
        """
        

        Parameters
        ----------
        variables : list, optional
            DESCRIPTION. The default is ["s", "p", "rho"].
        fig : TYPE, optional
            DESCRIPTION. The default is None.
        axs : TYPE, optional
            DESCRIPTION. The default is None.
        absolute : TYPE, optional
            If True, the profiles are plotted as they are. Else, they are 
            compared to the ones from the supplied external 1D model. 
            The default is True.
        external : tuple or list, optional
            depth and (Vs, Vp, rho) profiles from 1D model. The value is 
            automatically set  to None if the parameter "absolute" = True. 
            The default is None.
        interp_kwargs : dict, optional
            DESCRIPTION. The default is {}.
        third_variable : str or float or int, optional
            If not None, use this variable to color the line. Accepted values:
                - "T", str to plot as function of Temperature;
                - float, to plot as function of mean T at that depth;
                - integers, indicating the composition. 
            The default is None.

        Raises
        ------
        ValueError
            Will raise an error if absolute=+False and the external
            model supplied does not conform to [z, (, ,)] or (z, (, ,))

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        axs : TYPE
            DESCRIPTION.

        """
        if absolute:
            external = None
                    
        nv = len(variables)
        r_prof_km  = self.get_rprofile("s")[0] * self.r_E_km
        zprof_km = (self.r_E_km - r_prof_km)
        
        labels = _create_labels(variables, absolute)
        _profs = [self.get_rprofile(v)[-1] for v in variables]
        if external is None:
            profs = _profs
        else:
            if isinstance(external, (tuple, list)):
                ext_z, ext_profs = external
                _, profs = get_prof_pert(ext_z, ext_profs, zprof_km, _profs, 
                                         interp_kwargs)
            else:
                profs = None
                raise ValueError("External must be of type list or tuple, " +
                                 "the first element being the z coordinate " + 
                                 "in km and the second a tuple with s, p, rho")
        if axs is fig is None:
            fig, axs = plt.subplots(1, nv, sharey=True, squeeze=False)
            if axs.shape == (1,1):
                axs = [axs[1,1]]
        
        handle_mod = [None]
        
            
        for ax, prof in zip(axs, profs):
            if third_variable is None:
                handle_mod = ax.plot(prof, zprof_km, **plot_kwargs)
            else:
                cmap = cmap_kwargs["cmap"]
                vmin = cmap_kwargs["vmin"]
                vmax = cmap_kwargs["vmax"] 
                if third_variable == "T":
                    vals = self.get_rprofile("T")[-1]
                    # copied from matplotlib gallery (multicolored_line)
                    points = np.array([prof, zprof_km]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], 
                                              axis=1)
                    
                    norm = plt.Normalize(vmin, vmax)
                    lc = LineCollection(segments, cmap=cmap, norm=norm)
                    # Set the values used for colormapping
                    lc.set_array(vals)
                    lc.set_linewidth(1)
                    line = ax.add_collection(lc)
                    # fig.colorbar(line, ax=ax)
                elif isinstance(third_variable, float):
                    vals = self.get_rprofile("T")[-1]
                    val = np.interp(third_variable, 
                                    zprof_km[::-1], vals[::-1])
                    # print(val)
                    val -= vmin
                    val /= (vmax-vmin)
                    color = cm.get_cmap(cmap)(val)
                    plot_kwargs["color"] = color
                    handle_mod = ax.plot(prof, zprof_km, **plot_kwargs)
                else:
                    msg = "Composition as color of the line not implemented"
                    raise NotImplementedError(msg)
                    
        
        [ax.set_ylim((zprof_km.max(), zprof_km.min())) for ax in axs]
        [ax.set_xlabel(l) for ax, l in zip(axs, labels)]
        axs[0].set_ylabel("Depth [km]")
        axs[-1].legend(handle_mod, ["Model"])
        plt.subplots_adjust(wspace=0)
        plt.title(self.model_name)
        
        return fig, axs
    
    @staticmethod
    def import_hetfile(vel_model_path, variabs=["r", "theta", "p", "s", "rho"], 
                       fname="geodynamic_hetfile.sph"):
        
        name, extension = fname.rsplit(".")
        fpath = vel_model_path + "/" + fname
        
        if extension == "sph":  
            print("loadtxt")
            dic = {"r":0, "theta":1, "p":2, "s":3, "rho":4}
            usecols = [dic[v] for v in variabs]
            data = np.loadtxt(fpath, skiprows=1, usecols=usecols).T
        elif extension == "hdf5":
            data = []
            for d_name in variabs:
                with h5py.File(fpath, "r") as file:
                    d = np.array(file.get(d_name))
                    data.append(d)
            if isinstance(data, list) and len(data) < 2:
                data = data[0]
        else:
            data=None
        return data

    @staticmethod
    def plot_ext_prof(profs, axs, r_core_m=3481e3, r_Earth_m=6371e3, 
                      c="b", lbl=None):
        """
        

        Parameters
        ----------
        profs : str or list like
            Either the path to the 1D profiles or a tuple built like this:
                (depth_in_km, [s, p, rho]). 
        axs : TYPE
            DESCRIPTION.
        r_core_m : float, optional
            DESCRIPTION. The default is 3481e3.
        r_Earth_m : float, optional
            DESCRIPTION. The default is 6371e3.
        c : str, optional
            Color. The default is "b".
        lbl : str, optional
            label. The default is None.

        Returns
        -------
        None.

        """
        
        if isinstance(profs, str):
            zprem_km, _profs = get_ext_prof(profs, r_core_m, r_Earth_m)
        elif isinstance(profs, (list, tuple)):
            zprem_km, _profs = profs
            
        for ax, prof in zip(axs, _profs):
            handle = ax.plot(prof, zprem_km, c=c, label=lbl)
        axs[0].legend()
    
    def sh(self):
        pass