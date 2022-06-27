import numpy as np
import matplotlib.pyplot as plt

from chimera_project import Project
from velocity_model import VelocityModel
from functions import (initialize_vmodels, 
                       geodynamic_to_thermoelastic,
                       compute_vmodels)

# %% input 
# load project
proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)
# specify interpolation parameters
interpolation_parameters = dict(
                                p = 4,                  
                                tree_args = {"leafsize": 10},
                                query_args = {"r": 0.08, 
                                              "return_sorted": True}
                                )

# %% main
# 1
initialize_vmodels(proj, **interpolation_parameters)

# 2
geodynamic_to_thermoelastic(proj)

# 3
v_model_path = compute_vmodels(proj, True)


# %% plotting
vmod = VelocityModel.load(v_model_path[0])
stag_2_perp = np.sqrt(vmod.rho_stagyy / vmod.rho)

every_n = 100
x = vmod.x[::every_n]
y = vmod.y[::every_n]
vel = vmod.s[::every_n] * stag_2_perp[::every_n] # useful 2 convert btwn them

plt.figure()
plt.title("cartesian coordinates")
plt.tricontourf(x, y, vel, levels=512)

#plt.axis("tight")
#plt.axis("equal")

# %%
r, th = vmod.r[::every_n], vmod.theta[::every_n]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(th, r, s=1, c=vel)
ax.set_theta_offset(np.pi / 2.0)

