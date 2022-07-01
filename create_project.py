""" Here I show how to create a new project"""

import sys
sys.path.append('..')

from chimera_project import Project

# %% 
proj = Project()
proj.test_mode_on = True 
# directory of project
proj.chimera_project_path = "/home/matteo/chimera-projects/"

# define stagyy models location and model names
proj.stagyy_path = "/home/matteo/stagyyRuns/ghostresults/"
proj.stagyy_model_names = ["BS_drhoLM350_z1_PrLM100"] 

# define perplex paths, where tab files are located
proj.perplex_path = "/home/matteo/PerpleX_scripts/tab-files/"

# % define axisem's mesh numpy path
proj.bg_model = "PREM_ISO_2s"

# thermodynamic variables
proj.thermo_var_names = ["T", "p_s"]           # as read by stagyy
# compositional fields
stagyy_field_names = ["bs", "hz"]      # as read by stagyy
# the corresponding perplex projects, order must match with stagyy
perplex_proj_names = ["bsXu08_1", "hzXu08_1"]

proj.c_field_names = [stagyy_field_names, perplex_proj_names]

# define time span in billion years
proj.time_span_Gy = [4.5]


# finally create project with the name
proj_name = "Checkerboard2"
proj.new(proj_name)