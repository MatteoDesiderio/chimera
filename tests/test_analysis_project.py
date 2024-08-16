#!/usr/bin/env python3
"""
Created on Fri Aug 16 11:35:02 2024

@author: matteo
"""

from chimera.chimera_project import Project


def test_create(project_path, input_data_dir, thermo_data_description):
    temporary_path = project_path.as_posix()

    proj = Project()
    proj.test_mode_on = False
    proj.quick_mode_on = True
    proj.chimera_project_path = f"{temporary_path}/"
    proj.stagyy_path = f"{input_data_dir}/"
    proj.stagyy_model_names = ["stagyyModel"]
    proj.thermo_data_path = f"{project_path}/ThermoData/"
    proj.thermo_data_names = [thermo_data_description]
    proj.bg_model = "no1DBackGroundModel"
    proj.time_span_Gy = [3]
    proj_name = "temporary_project"
    proj.new(proj_name)
