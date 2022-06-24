#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:33:42 2022

@author: matteo
"""

import sys
import numpy as np
from stagpy import stagyydata
import matplotlib.pyplot as plt

sys.path.append('..')

from field import Field
from velocity_model import VelocityModel
from utils import set_renormalized_fields
from interfaces.stag import loader
from chimera_project import Project

# load project
proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)

# interpolation parameters
p = 4                     # parameter of shepherds inv dist weight interp
# kdtree parameters for tree creation
tree_args = {"leafsize": 10}
# kdtree parameters for fast closest neighbour. r is the max radius for search
query_args = {"r": 0.08, 
              "return_sorted": True}

# %% load axisem high resolution grid 
path_modx, path_mody = proj.get_paths()
x = np.load(path_modx)
y = np.load(path_mody)
n = len(x)

# %% load data from stagyy with stagpy into fields
# TODO speed up the process by loading coords and fields at the same time
print("Initializing variables and compositional fields")
variables = [Field(v) for v in proj.thermo_var_names]
fields = [Field(v) for v in proj.c_field_names[0]]

print("Variables:", *proj.thermo_var_names)
print("Compositional Fields:", *proj.c_field_names[0])
print()

for model_name in proj.stagyy_model_names:
    print("Loading stagyy model:", model_name)
    parent_path = proj.chimera_project_path + proj.project_name + "/" 
    model_path = parent_path + model_name
    print("Data will be saved in", model_path)
    indices, years = proj.t_indices[model_name], proj.time_span_Gy
    print("for each of time steps in", years, 
          "Gy, corresponding to indices", indices)
    
    for i_t, t in zip(indices, years):
        sdat = stagyydata.StagyyData(proj.stagyy_path + model_name)

        for v in variables:
            v.coords = loader.load_coords(sdat)
            v.values = loader.load_field(sdat, v.name, i_t)

        for f in fields:
            f.coords = loader.load_coords(sdat)
            f.values = loader.load_field(sdat, f.name, i_t)

        # renormalize compositional fields
        set_renormalized_fields(*fields)
        
        print("Initializing velocity model for", t, "Gy")
        v_model = VelocityModel(model_name, i_t, t, proj.c_field_names[0])
        # interpolating stagyy fields on larger axisem grid
        print("Interpolating stagyy variables and fields on axisem mesh")
        v_model.T = variables[0].interpolate(x, y, p, tree_args, query_args)
        v_model.P = variables[1].interpolate(x, y, p, tree_args, query_args)

        for i, f in enumerate(fields):
            v_model.C[i] = f.interpolate(x, y, p, tree_args, query_args)
        
        snap_path = model_path + "/{}/".format(i_t)
        print("Saving velocity model for", t, "Gy in", snap_path )
        v_model.save(snap_path)
        print("Done")
        print()
