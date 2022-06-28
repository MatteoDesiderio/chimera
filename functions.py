import numpy as np
from stagpy import stagyydata

from field import Field
from velocity_model import VelocityModel
from utils import set_renormalized_fields

from interfaces.stag import loader
from interfaces.perp.tab import Tab
from interfaces.perp.thermo_elastic_field import ThermoElasticField

def initialize_vmodels(proj, p, tree_args, query_args):
    # load axisem high resolution grid 
    path_modx, path_mody = proj.get_paths()
    x = np.load(path_modx)
    y = np.load(path_mody)
    n = len(x)
    
    # load data from stagyy with stagpy into fields
    # TODO speed up the process by loading coords and fields at the same time
    print("Initializing variables and compositional fields")
    variables = [Field(v) for v in proj.thermo_var_names]
    fields = [Field(v) for v in proj.c_field_names[0]]
    rho_stagyy = Field("rho")
        
    print("Variables:", *proj.thermo_var_names)
    print("Compositional Fields:", *proj.c_field_names[0])
    print()
    
    for model_name in proj.stagyy_model_names:
        print("Loading stagyy model:", model_name)
        parent_path = proj.chimera_project_path + proj.project_name + "/" 
        model_path = parent_path + model_name
        print("Data will be saved in", model_path)
        indices, years = proj.t_indices[model_name], proj.time_span_Gy
        print("for each of time steps in", years, 
              "Gy, corresponding to indices", indices)
        
        for i_t, t in zip(indices, years):
            print("Loading coords and values with stagpy")
            sdat = stagyydata.StagyyData(proj.stagyy_path + model_name)
    
            for v in variables:
                v.coords = loader.load_coords(sdat)
                v.values = loader.load_field(sdat, v.name, i_t)
                
            rho_stagyy.coords = loader.load_coords(sdat)
            rho_stagyy.values = loader.load_field(sdat, rho_stagyy.name, i_t)
    
            for f in fields:
                f.coords = loader.load_coords(sdat)
                f.values = loader.load_field(sdat, f.name, i_t)
            
    
            # renormalize compositional fields
            set_renormalized_fields(*fields)
            
            print("Initializing velocity model for", t, "Gy")
            v_model = VelocityModel(model_name, i_t, t, x, y, 
                                    proj.c_field_names[0])
            # interpolating stagyy fields on larger axisem grid
            print("Interpolating stagyy variables and fields on axisem mesh")
            v_model.T = variables[0].interpolate(x, y, p, 
                                                 tree_args, query_args)
            v_model.P = variables[1].interpolate(x, y, p, 
                                                 tree_args, query_args)
            v_model.rho_stagyy = rho_stagyy.interpolate(x, y, p, 
                                                 tree_args, query_args)
    
            for i, f in enumerate(fields):
                v_model.C[i] = f.interpolate(x, y, p, tree_args, query_args)
            
            snap_path = model_path + "/{}/".format(i_t)
            v_path = snap_path + proj.vel_model_path
            print()
            print("Saving velocity model for", t, "Gy in", v_path )
            v_model.save(v_path)
            print("Done")
            print()
            
def geodynamic_to_thermoelastic(proj):   
    for model_name in proj.stagyy_model_names:
        parent_path = proj.chimera_project_path + proj.project_name + "/" 
        model_path = parent_path + model_name
        print("Compute thermoelastic properties for geodynamic model", 
              model_path)
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
            print("----------------------------------------------------------")
            print()

def compute_vmodels(proj, use_stagyy_rho=False): 
    v_model_paths = []
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
            v_model.compute_velocities(use_stagyy_rho)
            print("Overwriting velocity model saved in", v_path)
            v_model_paths.append(v_path)
            v_model.save(v_path)
            
            print("Done")
            print("----------------------------------------------------------")
            print()
            
    return v_model_paths
 
    
def export_vmodels(proj):
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
            v_model.export(v_path)
            print("Exporting to sph format")
            print("Done")
            print("----------------------------------------------------------")
            print()
 
    
 
    
 