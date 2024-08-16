#!/usr/bin/env python3

from chimera_project import Project
from functions import (
    compute_vmodels,
    export_vmodels,
    geodynamic_to_thermoelastic,
    initialize_vmodels,
)

# %% input
# load project
name = "ExampleProject/"


proj_path = "./exampleOutput/"
proj = Project.load(proj_path + name)

# %% initialize, interpolate
initialize_vmodels(proj, interp_type="linear")

# %% extract thermoelastic fields
geodynamic_to_thermoelastic(proj)

# %% seismic velocity
v_model_paths = compute_vmodels(proj, use_stagyy_rho=False)

# %% export
export_vmodels(proj, absolute=False, fname="geodynamic_hetfile.hdf5")
