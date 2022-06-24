import sys
import numpy as np
from stagpy import stagyydata
import matplotlib.pyplot as plt

sys.path.append('..')

from field import Field
from velocity_model import VelocityModel
from utils import set_renormalized_fields
from interfaces.perp.tab import Tab
from interfaces.perp.thermo_elastic_field import ThermoElasticField
from interfaces.stag import loader
from chimera_project import Project

# load project
proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)

# %% interpolation parameters
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
variables = [Field(v) for v in proj.thermo_var_names]
fields = [Field(v) for v in proj.c_field_names]

for model_name in proj.stagyy_model_names:
    model_path = proj.chimera_project_path + proj.project_name + model_name
    indices, years = proj.t_indices[model_name], proj.time_span_Gy
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

        v_model = VelocityModel(model_name, proj.c_field_names)
        
        snap_path = model_path + "{}/".format(i_t)

        # interpolating stagyy fields on larger axisem grid
        v_model.T = variables[0].interpolate(x, y, p, tree_args, query_args)
        v_model.P = variables[1].interpolate(x, y, p, tree_args, query_args)
        
        for i, f in enumerate(fields):
            v_model.C[i] = f.interpolate(x, y, p, tree_args, query_args)
    
# %% saving moduli and rho from PerpleX table in an edible format (npy)
# TODO speed up with numba (tab load especially)

        tabs = []
        thermo_fields = []
        T, P = v_model.T, v_model.P                            
        for i, f in enumerate(proj.c_field_names):
            inpfl = proj.perplex_path + proj.proj_names_dict[f] + ".tab" 
            tab = Tab(inpfl)                                   
            tab.load()                                        
            tab.remove_nans()                                  
            thermo_field = ThermoElasticField(tab, f)  
            thermo_field.extract(T, P, model_name)     
            thermo_field.save(snap_path + proj.elastic_path)
            tabs.append(tab)
            thermo_fields.append(thermo_field)

        v_model.average(*v_model.load_moduli(snap_path + proj.elastic_path, 
                                             proj.proj_names_dict))
        v_model.compute_velocities()
        v_model.save(snap_path + proj.vel_model_path)

# %%
plt.figure()
plt.tricontourf(x[::100], y[::100], v_model.s[::100], levels=512)
plt.axis("tight")
plt.axis("equal")

