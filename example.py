import numpy as np
import matplotlib.pyplot as plt

from chimera_project import (Project,
                             initialize_vmodels, extract_thermoelastic,
                             compute_vmodels)
from velocity_model import VelocityModel

# %%
# load project
proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)

interpolation_parameters = dict(
                                p = 4,                  
                                tree_args = {"leafsize": 10},
                                query_args = {"r": 0.08, 
                                              "return_sorted": True}
                                )

initialize_vmodels(proj, **interpolation_parameters)
extract_thermoelastic(proj)
v_model_paths = compute_vmodels(proj)

v_model = VelocityModel(v_model_paths[0])

# %%
plt.figure()
plt.tricontourf(x[::100], y[::100], v_model.s[::100], levels=512)
plt.axis("tight")
plt.axis("equal")

