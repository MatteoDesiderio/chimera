import os
import pickle

class ThermoData:
    def __init__(self):
        """
        Initialize Thermochemical Data 

        Returns
        -------
        None.

        """
        self.perplex_path = ""                              # path to tab files
        self.stagyy_path = ""                      # path to your stagyy models 
        self.thermo_var_names = []                          # as read by stagyy
        # as read by stagyy + corresponding perplex projects, order must match!
        self.c_field_names = [[], []]         
        self.elastic_path = "/elastic-fields/"
        self.description = ""     # title to describe the set of tab files used 
                
    @property
    def c_field_names(self):
        return self._c_field_names
    @c_field_names.setter
    def c_field_names(self, val):
        self._c_field_names = val
        cstagyy, cperplex = val
        self.proj_names_dict = {k:v for k, v in zip(cstagyy, cperplex)}
        
    def save(self, save_path):
        """
        Save a thermo_data somewhere

        Parameters
        ----------
        save_path : Str
            Path where the class is saved.

        Returns
        -------
        None.

        """
        parent = save_path + "/ThermoData/"
        os.mkdir(parent)
                
        with open(parent + self.description +'.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    