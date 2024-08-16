#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:08:51 2024

@author: matteo
"""
from numpy import loadtxt
from numpy import all as np_all

from chimera.chimera_project import Project
from chimera.functions import (initialize_vmodels,
                               geodynamic_to_thermoelastic,
                               compute_vmodels, export_vmodels)


    
def test_compute(project_path, thermo_data_description, input_data_dir):
    temporary_path = project_path.as_posix()

    project = Project.load(f"{temporary_path}/temporary_project/")
    
    initialize_vmodels(project, interp_type="linear")
    geodynamic_to_thermoelastic(project)
    _ = compute_vmodels(project, use_stagyy_rho=False)
    
    fname_sph = "geodynamic_hetfile.sph"
    export_vmodels(project, absolute=False, fname=fname_sph)

    
def test_correct_result(project_path, thermo_data_description, 
                        input_data_dir, ground_truth_vel_anomaly_dir):
    temporary_path = project_path.as_posix()

    project = Project.load(f"{temporary_path}/temporary_project/")
    new_sph_path = (f"{project.chimera_project_path}/temporary_project/" + 
                     "/stagyyModel/2/seism_vel-fields/")
    
    
    fname_sph = "geodynamic_hetfile.sph"
    print(f"{new_sph_path}/{fname_sph}")
    print("|" * 70)
    newSph = loadtxt(f"{new_sph_path}/{fname_sph}",
                     skiprows=1)
    truthSph = loadtxt(f"{ground_truth_vel_anomaly_dir}/{fname_sph}",
                       skiprows=1)
    
    assert(np_all(newSph == truthSph))
    
    
    

