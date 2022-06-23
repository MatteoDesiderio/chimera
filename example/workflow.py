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

# %% define stagyy model
model_name = "BS_drhoLM350_z1_PrLM100"

# %% define perplex paths
perp_path = "/home/matteo/PerpleX_scripts/tab-files/"
stag_path = "/home/matteo/stagyyRuns/ghostresults/"

# %% define axisem's mesh numpy path
axi_path = "./axisem_fields/"
axi_name = "PREM_ISO_2s"

# %% thermodynamic variables and compositional fields
var_names = ["T", "p_s"]                        # as read by stagyy
field_names = ["bs", "hz", "prim"]              # as read by stagyy
# the corresponding perplex projects, order must match
proj_names = ["bsXu08_1", "hzXu08_1", "primBr100_1"]

# %% tie perplex tab file names to our composition of choice
proj_names_dict = {k:v for k,v in zip(field_names, proj_names)}

# %% interpolation parameters
p = 4                     # parameter of shepherds inv dist weight interp
# kdtree parameters tree creation
tree_args = {"leafsize": 10}
# kdtree parameters for fast closest neighbour 
query_args = {"r": 0.08, 
              "return_sorted": True}

# %% Output paths 
# where are you going to save your thermoelastic fields for each composition?
thermo_path = "./elastic-fields/"
# where are you going to save your velocity fields?
vel_model_path = "./seism_vel-fields/"  

# %% load axisem high resolution grid 
x = np.load(axi_path + axi_name + "_x.npy")
y = np.load(axi_path + axi_name + "_y.npy")
n = len(x)

# %% initialize variables and fields 
variables = [Field(v) for v in var_names]
fields = [Field(v) for v in field_names]

# %% load data from stagyy with stagpy into fields
# TODO speed up the process by loading coords and fields ath same time
sdat = stagyydata.StagyyData(stag_path + model_name)
i_t = -100

for v in variables:
    v.coords = loader.load_coords(sdat)    
    v.values = loader.load_field(sdat, v.name, i_t) 

for f in fields:
    f.coords = loader.load_coords(sdat)    
    f.values = loader.load_field(sdat, f.name, i_t)

# renormalize compositional fields
set_renormalized_fields(*fields)

# %% interpolating stagyy fields on larger axisem grid
# TODO load the T, P, c directly into the class VelocityModel (class attr?)
v_array = np.empty((len(variables), n))
f_array = np.empty((len(fields), n))

for i, v in enumerate(variables):
    v_array[i] = v.interpolate(x, y, p, tree_args, query_args)
    
for i, f in enumerate(fields):
    f_array[i] = f.interpolate(x, y, p, tree_args, query_args)
    
# %% saving moduli and rho from PerpleX table in an edible format (npy)
# TODO speed up with numba (tab load especially)
print("Saving the moduli for each composition")
tabs = []
thermo_fields = []
T, P = v_array                                 # thermodyamic variables     
for i, f in enumerate(field_names):
    inpfl = perp_path + proj_names_dict[f] + ".tab"    # where are the tables
    tab = Tab(inpfl)                                   # initialize table
    tab.load()                                         # load values
    tab.remove_nans()                                  # sanitize
    thermo_field = ThermoElasticField(tab, f)  # initialize thermoelastic field
    thermo_field.extract(T, P, model_name)     # create/save thermoel field
    thermo_field.save(thermo_path + model_name)
    tabs.append(tab)
    thermo_fields.append(thermo_field)

# %%
v_model = VelocityModel(model_name)
v_model.average(thermo_path, f_array, 
                field_names, proj_names_dict)

v_model.compute_velocities()
v_model.save(vel_model_path)

# %%
plt.figure()
plt.scatter(x, y, c=v_model.s, s=1)
plt.axis("tight")
plt.axis("equal")

