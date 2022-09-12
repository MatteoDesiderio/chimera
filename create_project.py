""" Here I show how to create a new project"""

import sys
sys.path.append('..')

from chimera_project import Project

# %% 
proj = Project()
proj.test_mode_on = False 
# directory of project
proj.chimera_project_path = "/home/matteo/chimera-projects/"

# define stagyy models location and model names
proj.stagyy_path = "/media/matteo/seagate_external/"
proj.stagyy_model_names = ["BS_drhoLM350_z1_PrLM100"] 

# define perplex paths, where tab files are located
proj.thermo_data_path = "/home/matteo/chimera-projects/ThermoData/"

# % define axisem's mesh numpy path
proj.bg_model = "PREM_ISO_2s"


# define time span in billion years
proj.time_span_Gy = [4.5]


# finally create project with the name
proj_name = "PlumPudding"
proj.new(proj_name)