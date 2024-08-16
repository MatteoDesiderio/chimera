"""
This is the first step before creating the project
"""
from thermo_data import ThermoData

# %% List of parameters
thermo_var_names = ["T", "p_s"]                             # as read by stagyy
stagyy_field_names = ["bs", "hz", "prim"]                   # as read by stagyy

# can supply more than one set of tab files
# the order of th perplex tab files match with the order of stagyy comp fields
dataset = dict(description = "ExamplePerplexTables",

               perplex_proj_names = ["bsChimera_1",
                                     "hzChimera_1",
                                     "primChimera_1"],

               perplex_path = "./inputData/exampleTabFiles/")


save_path = "./exampleOutput/"

# %% Fill in thermochemical data
# initialize it
thermodata = ThermoData()
# give it a title
thermodata.description = dataset["description"]
# define perplex paths, where tab files are located
thermodata.perplex_path = dataset["perplex_path"]

# thermodynamic variables
thermodata.thermo_var_names = thermo_var_names
# compositional fields (must be careful)
thermodata.c_field_names = [stagyy_field_names,
                            dataset["perplex_proj_names"]]
thermodata.import_tab()
# finally create thermodata file with the name
thermodata.save(save_path)
