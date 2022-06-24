# load project
import sys

sys.path.append('..')
from velocity_model import VelocityModel
from chimera_project import Project

proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)


for model_name in proj.stagyy_model_names:
    parent_path = proj.chimera_project_path + proj.project_name + "/" 
    model_path = parent_path + model_name
    print("Loading velocity models from", model_path)
    indices, years = proj.t_indices[model_name], proj.time_span_Gy
    print("for each of time steps in", years, 
          "Gy, corresponding to indices", indices)
    print()
    for i_t, t in zip(indices, years):
        snap_path = model_path + "/{}/".format(i_t)
        v_path = snap_path + proj.vel_model_path
        print("Loading velocity model saved in", v_path)
        v_model = VelocityModel.load(v_path)
        
        moduli_location = snap_path + proj.elastic_path
        print("Averaging rho, K, G")
        v_model.average(*v_model.load_moduli(moduli_location, 
                                             proj.proj_names_dict))
        
        print("Computing seismic velocities")
        v_model.compute_velocities()
        print("Overwriting velocity model saved in", v_path)
        v_model.save(v_path)
        
        print("Done")
        print("--------------------------------------------------------------")
        print()