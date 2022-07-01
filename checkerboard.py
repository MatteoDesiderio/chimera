#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:00:05 2022

@author: matteo
"""
import numpy as np
import matplotlib.pyplot as plt

from chimera_project import Project
from velocity_model import VelocityModel
from functions import (initialize_vmodels,
                       geodynamic_to_thermoelastic,
                       compute_vmodels, export_vmodels)

from interfaces.stag  import loader
from stagpy import stagyydata
from utils import to_cartesian

# %% input
# load project
proj_path = "/home/matteo/chimera-projects/Checkerboard2/"
proj = Project.load(proj_path)
# specify interpolation parameters
interpolation_parameters = dict(
    interp_type="bilinear", # closest, bilinear, inv_dist_weight
    p=4,
    tree_args={"leafsize": 10},
    query_args={"r": 0.0}
)


checker_board_params = {"checker_board_params" : [1.0, 1.0,  # ampl 1 and 2
                                                  60, 5,    # freq and 2
                                                  1600.0, 3100.0]# tmin max
                        }

path = proj.stagyy_path + proj.stagyy_model_names[0]
xstag, ystag = to_cartesian(*loader.load_coords(stagyydata.StagyyData(path)))

# %% initialize, interpolate
params = {}
params.update( checker_board_params)
params.update(interpolation_parameters)

initialize_vmodels(proj, **params)

# %% extract thermoelastic fields
geodynamic_to_thermoelastic(proj)

# %% seismic velocity + export 
v_model_paths = compute_vmodels(proj, False)
export_vmodels(proj)

# %% loading one model
vmod = VelocityModel.load(v_model_paths[0])

# %% plot profile
sa, s_prof = vmod.anomaly("s")
pa, p_prof = vmod.anomaly("p")
rhoa, rho_prof = vmod.anomaly("rho")
r_prof = s_prof["r"] * vmod.r_E_km * 1e3

zprof_km = (6371e3-r_prof) / 1e3


fig, axs = plt.subplots(1,3, sharey=True)
(ax1,ax2,ax3) = axs
labels= ["Vp [m/s]", "Vs [m/s]", "rho [kg/m^3]"]

ax1.plot(p_prof["val"], zprof_km, label="Model")
ax2.plot(s_prof["val"], zprof_km, label="Model")
ax3.plot(rho_prof["val"], zprof_km, label="Model")

[ax.legend for ax in axs]
[ax.set_ylim(ax.get_ylim()[::-1]) for ax in axs]
[ax.set_xlabel(l) for ax, l in zip(axs, labels)]
ax1.set_ylabel("Depth [m]")
ax3.legend()
plt.subplots_adjust(wspace=0)

# %% plot radial anomaly
plt.figure()
ax = plt.gca()
ax.set_title("Vs anomaly")
ax.scatter(vmod.x, vmod.y,  c=sa, cmap="RdBu")

#ax.plot(xstag/xstag.max(), ystag/ystag.max(), 'k.', ms=.5)


# %% plot field
plt.figure()
n = 500
x, y = vmod.x[::n], vmod.y[::n]
v = vmod.T[::n]
plt.tricontourf(x, y, v, levels=512, cmap="RdBu")






