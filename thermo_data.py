from interfaces.perp.tab import Tab
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
        self.thermo_var_names = []                          # as read by stagyy
        # c fields of stagyy + corresponding perplex tables, order must match!
        self.c_field_names = [[], []]         
        self.elastic_path = "/elastic-fields/"
        self.description = ""     # title to describe the set of tab files used
        self.tabs = [] 
        self.proj_names_dict = {}  # a dictionary associating
                
    @property
    def c_field_names(self):
        return self._c_field_names
    @c_field_names.setter
    def c_field_names(self, val):
        self._c_field_names = val
        cstagyy, cperplex = val
        self.proj_names_dict = {k:v for k, v in zip(cstagyy, cperplex)}
    
    def import_tab(self):
        self.tab_files = [] 
        for f, tb in zip(*self.c_field_names):
            inpfl = self.perplex_path + tb + ".tab" 
            tab = Tab(inpfl)                                   
            tab.load()                                        
            tab.remove_nans()     
            self.tabs.append(tab)
    
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
    
    @staticmethod
    def load(path):
        with open(path + '.pkl', 'rb') as f:
            pickled_class = pickle.load(f)
        return pickled_class