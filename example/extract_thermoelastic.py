# load project
import sys

sys.path.append('..')
from velocity_model import VelocityModel
from interfaces.perp.tab import Tab
from interfaces.perp.thermo_elastic_field import ThermoElasticField
from chimera_project import Project

proj_path = "/home/matteo/chimera-projects/Marble-vs-PlumPudding/"
proj = Project.load(proj_path)


for model_name in proj.stagyy_model_names:
    parent_path = proj.chimera_project_path + proj.project_name + "/" 
    model_path = parent_path + model_name
    print("Compute thermoelastic properties for geodynamic model", model_path)
    indices, years = proj.t_indices[model_name], proj.time_span_Gy
    print("for each of time steps in", years, 
          "Gy, corresponding to indices", indices)
    
    print()
    for i_t, t in zip(indices, years):
        snap_path = model_path + "/{}/".format(i_t)
        v_path = snap_path + proj.vel_model_path
        v_model = VelocityModel.load(v_path)
        print("Loading P, T from velocity model saved in", v_path)
        T, P = v_model.T, v_model.P                            
        for i, f in enumerate(proj.c_field_names[0]):
            inpfl = proj.perplex_path + proj.proj_names_dict[f] + ".tab" 
            tab = Tab(inpfl)                                   
            tab.load()                                        
            tab.remove_nans()                                  
            thermo_field = ThermoElasticField(tab, f)  
            thermo_field.extract(T, P, model_name)     
            thermo_field.save(snap_path + proj.elastic_path)
            
        print("Done")
        print("--------------------------------------------------------------")
        print()
