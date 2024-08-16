"""Here I show how to create a new project."""


from chimera_project import Project

# %%
proj = Project()
proj.test_mode_on = False
proj.quick_mode_on = True
# directory of project
proj.chimera_project_path = "./exampleOutput/"

# define stagyy models location and model names
proj.stagyy_path = "./inputData/"
proj.stagyy_model_names = ["stagyyModel"]

# define perplex paths, where tab files are located
proj.thermo_data_path = "./exampleOutput/ThermoData/"
proj.thermo_data_names = ["ExamplePerplexTables"] * len(proj.stagyy_model_names)

# % define axisem's mesh numpy path
proj.bg_model = "no1DBackGroundModel"

# define time span in billion years
proj.time_span_Gy = [3]

# finally create project with the name
proj_name = "ExampleProject"
proj.new(proj_name)
