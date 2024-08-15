#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:30:53 2024

@author: matteo
"""
import sys
import os

from chimera.thermo_data import ThermoData
from chimera.chimera_project import Project
from chimera.functions import (initialize_vmodels,
                               geodynamic_to_thermoelastic,
                               compute_vmodels, export_vmodels)

def test_create_thermo_data(tmp_path, input_data_dir):
    temporary_path = tmp_path.as_posix()
    perplex_path = f"{input_data_dir}/tabFiles/"
    
    # Fill in thermochemical data
    
    # initialize it
    thermodata = ThermoData()
    
    # give it a title
    thermodata.description = "ExamplePerplexTables"
    
    # define perplex paths, where tab files are located
    thermodata.perplex_path = perplex_path

    # thermodynamic variables
    thermodata.thermo_var_names = ["T", "p_s"]
    
    # compositional fields (must be careful the order matches)
    stagyy_field_names = ["bs", 
                          "hz", 
                          "prim"] 
    perplex_proj_names = ["bsChimera_1", 
                          "hzChimera_1", 
                          "primChimera_1"]
                  
    thermodata.c_field_names = [stagyy_field_names, 
                                perplex_proj_names]
    thermodata.import_tab()
    
    # finally create thermodata file with the name
    thermodata.save(temporary_path)
    assert os.path.exists(f"{temporary_path}/ThermoData")

    # TODO add more assert statements ()
    
    # ------------------------------------------------------------------------
    new_project = Project()
    new_project.test_mode_on = False 
    new_project.quick_mode_on = True
    # directory of project
    new_project.chimera_project_path = f"{temporary_path}/"

    # define stagyy models location and model names
    new_project.stagyy_path = input_data_dir
    new_project.stagyy_model_names = ['stagyyModel'] 

    # define perplex paths, where tab files are located
    new_project.thermo_data_path = f"{temporary_path}/ThermoData/"
    new_project.thermo_data_names = ["ExamplePerplexTables"]

    # % define axisem's mesh numpy path
    new_project.bg_model = "no1DBackGroundModel"

    # define time span in billion years
    new_project.time_span_Gy = [3]

    # finally create project with the name
    proj_name = "temporary_project"
    new_project.new(proj_name)
    
    # ------------------------------------------------------------------------    
      
    assert os.path.exists(f"{temporary_path}/{proj_name}")
    # TODO add more assert statements ()
    
    # load Project
    loaded_project = Project.load(f"{temporary_path}/{proj_name}/")

    # initialize, interpolate
    initialize_vmodels(loaded_project, interp_type="linear")

    # extract thermoelastic fields
    geodynamic_to_thermoelastic(loaded_project)

    #  seismic velocity
    v_model_paths = compute_vmodels(loaded_project, use_stagyy_rho=False)
    
    # export
    fname_sph = "geodynamic_hetfile.sph"
    fname_hdf5 = "geodynamic_hetfile.hdf5"
    export_vmodels(loaded_project, absolute=False, 
                   fname=fname_sph)
    export_vmodels(loaded_project, absolute=False, 
                   fname=fname_hdf5)
    
    
  
    
    