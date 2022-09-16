from glob import glob
from chimera_project import Project
from shutil import copy

SOLVER_path = "/home/matteo/axisem-9f0be2f/SOLVER/"
proj_path = "/home/matteo/chimera-projects/Plum_vs_Marble/"

snap_list = glob(proj_path + "/*/*/")


print("Please, copy any of the following commands:")
print()
for snap in snap_list:
    path_inparam = snap + "seism_vel-fields/inparam_hetero"    
    
    print("cp", path_inparam, SOLVER_path, "\\")
    print("./submit.csh", snap+"axisem_run")
    print("\n#"+ "-"*30)