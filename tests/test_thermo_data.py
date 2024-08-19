"""
Created on Thu Aug 15 19:30:53 2024.

@author: matteo
"""

import os

from chimera.thermo_data import ThermoData


def test_create(project_path, input_data_dir, thermo_data_description):
    temporary_path = project_path.as_posix()

    perplex_path = f"{input_data_dir}/tabFiles/"
    thermodata = ThermoData()
    thermodata.description = thermo_data_description
    thermodata.perplex_path = perplex_path
    thermodata.thermo_var_names = ["T", "p_s"]
    stagyy_field_names = ["bs", "hz", "prim"]
    perplex_proj_names = ["bsChimera_1", "hzChimera_1", "primChimera_1"]
    thermodata.c_field_names = [stagyy_field_names, perplex_proj_names]
    thermodata.import_tab()
    thermodata.save(temporary_path)

    pickle_path = f"{temporary_path}/ThermoData/{thermo_data_description}.pkl"
    assert os.path.exists(pickle_path)


def test_load(project_path, thermo_data_description):
    temporary_path = project_path.as_posix()
    pickle_path = f"{temporary_path}/ThermoData/{thermo_data_description}"

    loaded_thermo = ThermoData.load(pickle_path)

    assert isinstance(loaded_thermo, ThermoData)
