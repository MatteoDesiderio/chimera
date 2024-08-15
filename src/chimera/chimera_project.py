import pickle
import os
import warnings
import numpy as np
from stagpy import stagyydata
from stagpy.field import get_meshes_fld
from .velocity_model import VelocityModel
from .utils import rms

def _get_mesh_xy(sdat):
    #geom = sdat.snaps[0].geom
    mesh_x, mesh_y, _, _ = get_meshes_fld(sdat.snaps[0], "T")
    mesh_x = mesh_x.squeeze().flatten()
    mesh_y = mesh_y.squeeze().flatten()
    return mesh_x / mesh_x.max(), mesh_y / mesh_y.max()

class Project:
    """
    Project class
    """
    def __init__(self):
        """
        Initialize project 

        Returns
        -------
        None.

        """
        self.test_mode_on = False
        self.quick_mode_on = False
        self.custom_mesh_shape = None
        self._GY = 3600*24*364*1e9                                        # [s]
        self.chimera_project_path = ""    # path where you want to save project
        self.bg_model = ""                       # name of your axisem bg model
        self.thermo_data_path = ""           # parent path to thermo_data files
        self.thermo_data_names = ""       # their names. single str of list str 
        self.stagyy_path = ""                      # path to your stagyy models 
        self.stagyy_model_names = []
        self.elastic_path = "/elastic-fields/"
        self.vel_model_path = "/seism_vel-fields/" 
        self.time_span_Gy = [] # timesteps for which you want to compute vmodel
        self.geom = None      # geom of the geodynamic models grid (same 4 all)
        self._regular_rect_mesh = False

    @property
    def stagyy_model_names(self):
        return self._stagyy_model_names
    @stagyy_model_names.setter
    def stagyy_model_names(self, val):
        self._stagyy_model_names = val
        self.t_indices = {k:[] for k in self._stagyy_model_names}
    
    # for a quick analysis, we don't need an AxiSEM mesh to begin with
    @property
    def bg_model(self):
        return self._bg_model
    
    @bg_model.setter
    def bg_model(self, val):
        if self.quick_mode_on is None:
            self._bg_model = None
        else:
            self._bg_model = val
            
    # if you are using the StagYY grid, you don't need a custom grid 
    @property
    def custom_mesh_shape(self):
        return self._custom_mesh_shape
    @custom_mesh_shape.setter
    def custom_mesh_shape(self, val):
        if self.quick_mode_on:
            self._custom_mesh_shape = None
            msg = "Quick mode activated, but mesh shape provided:" + \
                  "custom_mesh_shape has been set to None."
            warnings.warn(msg)
        else:
            self._custom_mesh_shape = val
        
    def new(self, proj_name="New Project"):
        """
        Create a new project

        Parameters
        ----------
        proj_name : TYPE, optional
            DESCRIPTION. The default is "New Project".

        Returns
        -------
        None.

        """
        self.project_name = proj_name
        parent = self.chimera_project_path + proj_name + "/"
        os.mkdir(parent)
        geoms = []
        for nm in self.stagyy_model_names:
            print("Loading", nm)
            path = parent + nm
            os.mkdir(path)
            for t in self.time_span_Gy:
                if self.test_mode_on:
                    index = 0
                else:
                    sdat = stagyydata.StagyyData(self.stagyy_path + nm)
                    index = sdat.snaps.at_time(t * self._GY).isnap
                    geom = sdat.par["geometry"]
                path_snap = (path + "/%i") % index
                os.mkdir(path_snap)
                os.mkdir(path_snap + self.elastic_path)
                os.mkdir(path_snap + self.vel_model_path)
                self.t_indices[nm].append(index)
            geoms.append(geom)
            
        if not np.all(np.r_[geoms] ==  geom):
            msg = "All models must share the same geometry parameters"
            raise NotImplementedError(msg)
        else:
            self.geom = geoms[0]

        with open(parent + 'project_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def get_mesh_xy(self):
        """
        Generate x and y 

        
        -------
        mesh_x : TYPE
            DESCRIPTION.
        mesh_y : TYPE
            DESCRIPTION.

        """
        if self.quick_mode_on:
            print("We assume all models are defined on the same grid")
            nm = self.stagyy_model_names[0]
            sdat = stagyydata.StagyyData(self.stagyy_path + nm)
            mesh_x, mesh_y = _get_mesh_xy(sdat)
        else:
            path_x = self.chimera_project_path + self.bg_model + "_x.npy"
            path_y = self.chimera_project_path + self.bg_model + "_y.npy"
            mesh_x = np.load(path_x)
            mesh_y = np.load(path_y)
            if not self.custom_mesh_shape is None:
                try:
                    shp = self.custom_mesh_shape
                    np.reshape(mesh_x, shp)
                    np.reshape(mesh_y, shp)
                    self._regular_rect_mesh = True
                    msg = "succesfully reshaped mesh into provided shape." + \
                          "Please, check quality of result."
                    warnings.warn(msg)
                except ValueError:
                    self._regular_rect_mesh = False
                    self.custom_mesh_shape = None
                    msg = "cannot reshape mesh into provided shape." + \
                          "Continuing assuming non-rectangular, " + \
                          "axisem-like mesh: custom_mesh_shape set to None."
                    warnings.warn(msg)
                    
        return mesh_x, mesh_y
    

    @staticmethod
    def load_vel_models_by_year(proj):
        name_year_map = proj.t_indices
        project_path = proj.chimera_project_path + proj.project_name + \
                       "/{}/{}" + proj.vel_model_path
        velocity_models_dict = {} 
        for vel_model_name in name_year_map:
            t_indices = name_year_map[vel_model_name]
            models_list = []
            for t_index in t_indices:
                model_path = project_path.format(vel_model_name, t_index)
                vmod = VelocityModel.load(model_path)
                models_list.append(vmod)
            velocity_models_dict[vel_model_name] = models_list
        return (proj, velocity_models_dict)

    @staticmethod
    def compute_profiles(proj, velocity_models_dict):
        name_year_map = proj.t_indices
        vmod = velocity_models_dict[proj.stagyy_model_names[0]][0]
        shape = proj.geom["nytot"], proj.geom["nztot"]
        z = np.reshape((1 - vmod.r ) * vmod.r_E_km, shape)[0]
        
        heterogeneity_dictionary = {}
        for vel_model_name in name_year_map:
            vmods = velocity_models_dict[vel_model_name]
            t_indices = name_year_map[vel_model_name]
            
            spr = [m.get_rprofile("s")[-1] for m in vmods]
            ppr = [m.get_rprofile("p")[-1] for m in vmods]
            rho = [m.get_rprofile("rho")[-1] for m in vmods]
            b = [m.bulk.reshape(shape) for m in vmods]
            prim = [m.C[-1].reshape(shape) for m in vmods]
            T = [m.T.reshape(shape) for m in vmods]
            T_a = [m.anomaly("T")[0].reshape(shape) for m in vmods]
            s = [m.s.reshape(shape) for m in vmods]
            p = [m.p.reshape(shape) for m in vmods]
            s_a = [m.anomaly("s")[0].reshape(shape) for m in vmods]
            p_a = [m.anomaly("p")[0].reshape(shape) for m in vmods]
            b_a = [m.anomaly("bulk")[0].reshape(shape) for m in vmods]
            
            s_lim = 0.4
            p_lim = 0.2
            s_safe = [np.ma.masked_array(x, np.abs(x) < s_lim) for x in s_a]
            p_safe = [np.ma.masked_array(x, np.abs(x) < p_lim) for x in p_a]
            s_safe = [np.ma.masked_array(x, x * y < 0) 
                      for x, y in zip(s_safe, p_safe)]
            p_safe = [np.ma.masked_array(y, x * y < 0) 
                      for x, y in zip(s_safe, p_safe)]
            
            r_sp = [s / p for s, p in zip(s_safe, p_safe)]
            
            cor_b_s = [np.r_[[np.corrcoef(x[:,i], y[:,i])[0,1] 
                              for i in range(shape[1])]]  
                       for x, y in zip(b_a, s_a)]
            
            profiles_keys = "s_a", "p_a", "b_a", "b", "r_sp", "s", "p"
            profiles = {k:{} for k in profiles_keys}
            functions = [np.ma.mean, np.ma.median, np.ma.std, rms]
            fnames = ["mean", "median", "std", "rms"]
            for fun, fkey in zip(functions, fnames):
                for xx, key in zip([s_a, p_a, b_a, b, r_sp, s, p], 
                                   profiles_keys):
                    profiles[key][fkey] = [fun(x, axis=0) for x in xx]
            
            r_sp_robust = np.ma.vstack(profiles["s_a"]["rms"]) / \
                          np.ma.vstack(profiles["p_a"]["rms"])
            
            hets = dict(z=z, lims=({"s_lim": s_lim, "p_lim": p_lim}),
                        t_indices=t_indices, mod_name=vel_model_name,
                        prim=prim, profs=profiles, s=spr, p=ppr, rho=rho, 
                        cor_b_s=cor_b_s, r_sp_robust=r_sp_robust, T=T, T_a=T_a)
            
            heterogeneity_dictionary[vel_model_name] = hets
            
        return heterogeneity_dictionary
    
    @staticmethod
    def save_heterogeneity_dictionary(heterogeneity_dictionary, path):
        for vname in heterogeneity_dictionary:
            fname = "{}/{}_het_profs.pkl".format(path, vname)
            with open(fname , "wb") as f:
                pickle.dump(heterogeneity_dictionary[vname], f)
    
    @staticmethod
    def load(project_path):
        with open(project_path + 'project_data.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class
        
        
        
        