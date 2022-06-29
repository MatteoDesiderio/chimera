import numpy as np
import matplotlib.pyplot as plt

from chimera_project import Project
from velocity_model import VelocityModel
from functions import (initialize_vmodels,
                       geodynamic_to_thermoelastic,
                       compute_vmodels, export_vmodels)

# %% input
# load project
proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)
# specify interpolation parameters
interpolation_parameters = dict(
    interp_type="bilinear", # closest, bilinear, inv_dist_weight
    p=4,
    tree_args={"leafsize": 10},
    query_args={"r": 0.0}
)


# %% main
# 1
initialize_vmodels(proj, **interpolation_parameters)

# %%
# 2
geodynamic_to_thermoelastic(proj)

#%%
# 3
v_model_paths = compute_vmodels(proj, False)

# 4
export_vmodels(proj)

# %% loading the computed model
vmod = VelocityModel.load(v_model_paths[0])

# %%
stag_2_perp = np.sqrt(vmod.rho_stagyy / vmod.rho)

every_n = 100
x = vmod.x[::every_n]
y = vmod.y[::every_n]
# * stag_2_perp[::every_n] # useful 2 convert btwn them
vel = vmod.s[::every_n]

plt.figure()
plt.title("cartesian coordinates")
plt.tricontourf(x, y, vel, levels=512)

# %%
sa, s_prof = vmod.anomaly("s")
pa, p_prof = vmod.anomaly("p")
rhoa, rho_prof = vmod.anomaly("rho_stagyy")
r_prof = s_prof["r"] * vmod.r_E_km * 1e3

d = "/home/matteo/axisem-9f0be2f/SOLVER/MESHES/test_VERBOSE/1dmodel_axisem.bm"
rprem,rhoprem,vpprem,vsprem,_,_ = np.loadtxt(d, skiprows=6, unpack=True)
mantle = rprem >= 3481e3
rprem, rhoprem = rprem[mantle], rhoprem[mantle]
vpprem, vsprem = vpprem[mantle], vsprem[mantle] 

zprem_km = (6371e3-rprem) / 1e3
zprof_km = (6371e3-r_prof) / 1e3


fig, axs = plt.subplots(1,3, sharey=True)
(ax1,ax2,ax3) = axs
labels= ["Vp [m/s]", "Vs [m/s]", "rho [kg/m^3]"]

ax1.plot(vpprem, zprem_km, label="PREM")
ax2.plot(vsprem, zprem_km, label="PREM")
ax3.plot(rhoprem, zprem_km, label="PREM")

ax1.plot(p_prof["val"], zprof_km, label="Model")
ax2.plot(s_prof["val"], zprof_km, label="Model")
ax3.plot(rho_prof["val"], zprof_km, label="Model")

[ax.legend for ax in axs]
[ax.set_ylim(ax.get_ylim()[::-1]) for ax in axs]
[ax.set_xlabel(l) for ax, l in zip(axs, labels)]
ax1.set_ylabel("Depth [m]")
ax3.legend()
plt.subplots_adjust(wspace=0)

# %%
plt.figure()
plt.title("Vs anomaly")
plt.tricontourf(vmod.x, vmod.y, sa, levels=512, cmap="RdBu",
                vmin=-.05, vmax=.05)

plt.axis("tight")
plt.axis("equal")

# %%
r, th = vmod.r[::every_n], vmod.theta[::every_n]
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(th, r, s=1, c=vel)
ax.set_theta_offset(np.pi / 2.0)
