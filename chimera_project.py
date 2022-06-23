#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:14:07 2022

@author: matteo
"""
from interfaces.axi import mesh_importer as mi
import os 
import pickle
from models_collection import ModelsCollection
from stagpy import stagyydata
from numpy import savetxt

class Project:
    def __init__(self):
        self._GY = 3600*24*364*1e9
        self.chimera_project_path = "" # path where you want to save project
        self.project_name = "" # path containing your project
        self.bg_model = "" # name of your axisem bg model
        self.perplex_path = "" # path to tab files
        self.stagyy_path = "" # path to your stagyy models 
        self.axisem_path = "" # path to axisem 
        self.stagyy_model_names = []
        
        self.thermo_var_names = []                # as read by stagyy
        self.c_field_names = []           # as read by stagyy
        # the corresponding perplex projects, order must match
        self.perplex_proj_names = []
        self.proj_names_dict = {k: v for k, v in zip(self.c_field_names,
                                                     self.perplex_proj_names)}
        self.elastic_path = ""
        self.vel_model_path = "" 
        self.time_span_Gy = []       
        self.t_indices = {k:[] for k in self.stagyy_model_names}
        
    def create_project(self):
        os.mkdir(self.chimera_project_path + self.project_name)
        for nm in self.stagyy_model_names:
            path = self.chimera_project_path + self.project_name + nm
            os.mkdir(path)
            for t in self.time_span_Gy:
                sdat = stagyydata.StagyyData(self.stagyy_path + nm)
                index = sdat.snaps.at_time(t * self._GY).istep
                path_snap = (path + "/%i") % index
                os.mkdir(path_snap)
                self.t_indices[nm].append(index)
                
        with open('project_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
        
        
        