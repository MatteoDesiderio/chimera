#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:14:07 2022

@author: matteo
"""
from interfaces.axi import mesh_importer as mi
import os 
import pickle

class Project:
    def __init__(self):
        self.chimera_project_path = "" # path where you want to save project
        self.project_name = "" # path containing your project
        self.bg_model = "" # name of your axisem bg model
        self.perplex_path = "" # path to tab files
        self.stagyy_path = "" # path to your stagyy models 
        self.axisem_path = "" # path to axisem 
        self.stagyy_model_names = []
        
    def create_project(self):
        os.mkdir(self.chimera_project_path + self.project_name)
        for nm in self.stagyy_model_names:
            os.mkdir(self.chimera_project_path + self.project_name + nm)
            
        with open('project_data.pkl', 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        
        
        
        