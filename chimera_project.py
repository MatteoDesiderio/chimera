import pickle
import os
from stagpy import stagyydata
from stagpy.field import get_meshes_fld
import numpy as np

def _get_mesh_xy(sdat):
    geom = sdat.snaps[0].geom
    mesh_x = geom.x_mesh.squeeze().flatten()
    mesh_y = geom.y_mesh.squeeze().flatten()
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
        if self.quick_mode_on:
            self._bg_model = None
        else:
            self._bg_model = val
        
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

        with open(parent + 'project_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def get_mesh_xy(self):
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
        return mesh_x, mesh_y
            
    @staticmethod
    def load(project_path):
        with open(project_path + 'project_data.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class
        
        
        
        