import pickle
import os
from stagpy import stagyydata


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
        self._GY = 3600*24*364*1e9                                        # [s]
        self.chimera_project_path = ""    # path where you want to save project
        self.bg_model = ""                       # name of your axisem bg model
        self.perplex_path = ""                              # path to tab files
        self.stagyy_path = ""                      # path to your stagyy models 
        self.stagyy_model_names = []
        self.thermo_var_names = []                          # as read by stagyy
        # as read by stagyy + corresponding perplex projects, order must match!
        self.c_field_names = [[], []]         
        self.elastic_path = "/elastic-fields/"
        self.vel_model_path = "/seism_vel-fields/" 
        self.time_span_Gy = [] # timesteps for which you want to compute vmodel
                
    @property
    def c_field_names(self):
        return self._c_field_names
    @c_field_names.setter
    def c_field_names(self, val):
        self._c_field_names = val
        cstagyy, cperplex = val
        self.proj_names_dict = {k:v for k, v in zip(cstagyy,
                                                    cperplex)}
    @property
    def stagyy_model_names(self):
        return self._stagyy_model_names
    @stagyy_model_names.setter
    def stagyy_model_names(self, val):
        self._stagyy_model_names = val
        self.t_indices = {k:[] for k in self._stagyy_model_names}
        
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
        for nm in self.stagyy_model_names:
            path = parent + nm
            os.mkdir(path)
            for t in self.time_span_Gy:
                sdat = stagyydata.StagyyData(self.stagyy_path + nm)
                index = sdat.snaps.at_time(t * self._GY).isnap
                path_snap = (path + "/%i") % index
                os.mkdir(path_snap)
                os.mkdir(path_snap + self.elastic_path)
                os.mkdir(path_snap + self.vel_model_path)
                self.t_indices[nm].append(index)
                
        with open(parent + 'project_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def get_paths(self):
        mesh_x = self.chimera_project_path + self.bg_model + "_x.npy"
        mesh_y = self.chimera_project_path + self.bg_model + "_y.npy"
        return (mesh_x, mesh_y)
            
    @staticmethod
    def load(project_path):
        with open(project_path + 'project_data.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class
        
        
        
        