import sys
import numpy as np
from stagpy import stagyydata

sys.path.append('..')


from field import Field
from interfaces.perp.tab import Tab
from interfaces.perp.thermo_elastic_field import ThermoElasticField

from interfaces.stag import loader

# %% define perplex paths
perp_path = "/home/matteo/PerpleX_scripts/tab-files/"
stag_path = "/home/matteo/stagyyRuns/ghostresults/"
    
# %% load axisem high resolution grid 
x = np.load("../PREM_ISO_2s_x.npy")
y = np.load("../PREM_ISO_2s_y.npy")
n = len(x)

# %% define stagyy model
model_name = "BS_drhoLM350_z1_PrLM100"

# %% create fields
var_names = ["T", "p_s"] 
variables = [Field(v) for v in var_names]

field_names = ["bs", "hz", "prim"]
fields = [Field(v) for v in field_names]

# tie perplex tab file names to our composition of choice
proj_names = {'prim': 'primBr100_1',
              'hz': 'hzXu08_1',
              'bs': 'bsXu08_1'}

# %% load data from stagyy with stagpy into fields
sdat = stagyydata.StagyyData(stag_path + model_name)
i_t = -100

for v in variables:
    v.coords = loader.load_coords(sdat)    
    v.values = loader.load_field(sdat, v.name, i_t) 

for f in fields:
    f.coords = loader.load_coords(sdat)    
    f.values = loader.load_field(sdat, f.name, i_t)

# %% interpolating
v_array = np.empty((n, len(variables)))
f_array = np.empty((n, len(fields)))
p = 4
tree_args = {"leafsize": 10}
query_args = {"r": 0.08, 
              "return_sorted": True}

for i, v in enumerate(variables):
    v_array[:, i] = v.interpolate(x, y, p, tree_args, query_args)
    
for i, f in enumerate(fields):
    f_array[:, i] = f.interpolate(x, y, p, tree_args, query_args)
    
# %%

for f in field_names:
    inpfl = perp_path + proj_names[f] + ".tab"
    tab = Tab(inpfl)
    tab.load()
    tab.remove_nans()
    thermo_field = ThermoElasticField(tab, f)
    T, P = v_array.T
    thermo_field.extract(T, P, model_name)
    thermo_field.save("./numpy-files/" + model_name)
    
    