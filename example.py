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
v_model_paths = compute_vmodels(proj)


# %% plotting
v = VelocityModel.load(v_model_paths[0])


plt.figure()
plt.title("cartesian coordinates")
plt.tricontourf(v.x[::100], v.y[::100], v.s[::100], levels=512)

plt.axis("tight")
plt.axis("equal")


r, th = v.r[::100], v.theta[::100]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(th, r, s=1, c=v.s[::100])
ax.set_theta_offset(np.pi / 2.0)

