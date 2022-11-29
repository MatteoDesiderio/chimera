import os
import numpy as np
from stagpy import stagyydata

from field import Field
from velocity_model import VelocityModel
from utils import set_renormalized_fields

from interfaces.stag import loader
from interfaces.perp.tab import Tab
from interfaces.perp.thermo_elastic_field import ThermoElasticField

from thermo_data import ThermoData

def _checker(shape, 
             a1=1.0, a2=0.0, freq1=10, freq2=10, Tmin=1600.0, Tmax=3100.0):
    a1 = min(1.0, np.abs(a1))
    a2 = min(1.0, np.abs(a2))
    a = a1*np.sin( np.indices(shape)[0] / (shape[0] / freq1) )
    b = a2*np.cos( np.indices(shape)[1] / (shape[1] / freq2) )
    T_ = (a + b) / (a1 + a2)
    T_ = np.ceil(T_)
    T_[T_ == 0] = Tmin
    T_[T_ == 1] = Tmax
    return T_


def initialize_vmodels(proj, interp_type, checker_board_params=None):
    
    # load axisem high resolution grid 
    path_modx, path_mody = proj.get_paths()
    x = np.load(path_modx)
    y = np.load(path_mody)
    n = len(x)
    

    if proj.test_mode_on:
        if checker_board_params is None:
            checker_board_params = [1.0, 0.0, 10, 10, 1600.0, 3100.0]
            
        print("Test Mode: 'time_span_Gy' and 'stagyy_model_names' ",
              "were used but bear no meaning. Stagyy model used to load ", 
              "coordinates and P, but T field is a checkerboard, ",
              "while the comp field is all pyrolitic")
        for model_name, thermo_name in zip(proj.stagyy_model_names, 
                                           proj.thermo_data_names):
            
            print("Initializing variables and compositional fields")
            thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
            variables = [Field(v) for v in thermodata.thermo_var_names]
            fields = [Field(v) for v in thermodata.c_field_names[0]]
            rho_stagyy = Field("rho")
            print("Variables:", *thermodata.thermo_var_names)
            print("Compositional Fields:", *thermodata.c_field_names[0])
            print()
            
            parent_path = proj.chimera_project_path + proj.project_name + "/" 
            model_path = parent_path + model_name
            indices, years = proj.t_indices[model_name], proj.time_span_Gy
            
            for i_t, t in zip(indices, years):
                sdat = stagyydata.StagyyData(proj.stagyy_path + model_name)
                for v in variables:
                    v.coords = loader.load_coords(sdat)
                    shape = ( len(v.coords[1]), len(v.coords[0]) )
                    if v.name == "T":
                        v.values = _checker(shape, *checker_board_params)
                    else:
                        v.values = loader.load_field(sdat, v.name, i_t)
                
                for f in fields:
                    f.coords = loader.load_coords(sdat)
                    if f.name == "prim":
                        f.values = np.zeros(shape) 
                    elif f.name == "hz":
                        f.values = np.ones(shape) * .8 
                    elif f.name == "bs":
                        f.values = np.ones(shape) * .2
                    
                # renormalize compositional fields
                set_renormalized_fields(fields)
                
                v_model = VelocityModel(model_name, i_t, t, x, y, 
                                        proj.c_field_names[0])
                # interpolating stagyy fields on larger axisem grid

                v_model.T = variables[0].interpolate(interp_type, x, y)
                v_model.P = variables[1].interpolate(interp_type, x, y)
        
                for i, f in enumerate(fields):
                    v_model.C[i] = f.interpolate(interp_type, x, y)
                
                snap_path = model_path + "/{}/".format(i_t)
                v_path = snap_path + proj.vel_model_path
                print()
                print("Saving velocity model for", t, "Gy in", v_path )
                v_model.thermo_data_name = proj.thermo_data_path + thermo_name
                v_model.save(v_path)
                print("Done")
                print()
                
    else:
        zipnames = zip(proj.stagyy_model_names, proj.thermo_data_names)
        # load data from stagyy with stagpy into fields
        # TODO speed up the process by loading coords, fields at the same time
        for model_name, thermo_name in zipnames:
            thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
            lims = thermodata.range
            print("Initializing variables and compositional fields")            
            variables = [Field(v) for v in thermodata.thermo_var_names]
            fields = [Field(v) for v in thermodata.c_field_names[0]]
            rho_stagyy = Field("rho")
            
            print("Variables:", *thermodata.thermo_var_names)
            print("Compositional Fields:", *thermodata.c_field_names[0])
            print()
            
            print("Loading stagyy model:", model_name)
            parent_path = proj.chimera_project_path + proj.project_name + "/" 
            model_path = parent_path + model_name
            print("Data will be saved in", model_path)
            indices, years = proj.t_indices[model_name], proj.time_span_Gy
            print("for each of time steps in", years, 
                  "Gy, corresponding to indices", indices)
            
            print("The thermodynamic dataset '%s' will be used," % thermo_name)
            print("located in %s and containing reference to the tab files:" %
                  proj.thermo_data_path)
            print(*thermodata.c_field_names[1])
            print()
            for i_t, t in zip(indices, years):
                print("Loading coords and values with stagpy")
                sdat = stagyydata.StagyyData(proj.stagyy_path + model_name)
                
                for v, key in zip(variables, lims):
                    a_min, a_max = lims[key]
                    v.coords = loader.load_coords(sdat)
                    print("Clamping %s between %.1e and %.1e [SI units]" % 
                          (key, a_min, a_max) )
                    v.values = np.clip(loader.load_field(sdat, v.name, i_t),
                                       a_min, a_max) 
                    
                rho_stagyy.coords = loader.load_coords(sdat)
                rho_stagyy.values = loader.load_field(sdat, rho_stagyy.name, 
                                                      i_t)
        
                for f in fields:
                    f.coords = loader.load_coords(sdat)
                    f.values = loader.load_field(sdat, f.name, i_t)
                
                # renormalize compositional fields
                set_renormalized_fields(fields)
                
                print("Initializing velocity model for", t, "Gy")
                v_model = VelocityModel(model_name, i_t, t, x, y, 
                                        thermodata.c_field_names[0])
                # interpolating stagyy fields on larger axisem grid
                print("Interpolating stagyy variables",
                      "and fields on axisem mesh")
                v_model.T = variables[0].interpolate(interp_type, x, y)
                v_model.P = variables[1].interpolate(interp_type, x, y)
                v_model.rho_stagyy = rho_stagyy.interpolate(interp_type, x, y)
        
                for i, f in enumerate(fields):
                    v_model.C[i] = f.interpolate(interp_type, x, y)
                
                snap_path = model_path + "/{}/".format(i_t)
                v_path = snap_path + proj.vel_model_path
                print()
                print("Saving velocity model for", t, "Gy in", v_path )
                v_model.thermo_data_name = proj.thermo_data_path + thermo_name
                v_model.save(v_path)
                print("Done")
                print()
            
def geodynamic_to_thermoelastic(proj):
    """
    

    Parameters
    ----------
    proj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    zip_names = zip(proj.stagyy_model_names, proj.thermo_data_names)
    for model_name, thermo_name in zip_names:
        parent_path = proj.chimera_project_path + proj.project_name + "/" 
        model_path = parent_path + model_name
        print("Compute thermoelastic properties for the geodynamic model:") 
        print(model_path)
        print("which uses the thermodynamic data saved as:")
        print(proj.thermo_data_path + thermo_name)
        print()
        indices, years = proj.t_indices[model_name], proj.time_span_Gy
        print("Time steps: ", years, 
              "Gy, corresponding to indices", indices)
        print()
        thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
        for i_t, t in zip(indices, years):
            snap_path = model_path + "/{}/".format(i_t)
            v_path = snap_path + proj.vel_model_path
            v_model = VelocityModel.load(v_path)
            save_path = snap_path + proj.elastic_path
            
            # checking if there is anything missing 
            exist = []
            for tab, f in zip(thermodata.tabs, thermodata.c_field_names[0]):
                nm = save_path + tab.tab["title"]
                for v in ["rho", "K", "G"]:
                    exist.append(os.path.exists(nm + "_" + v + ".npy"))
            
            # conservatively, we will redo the look-up for all elastic
            # properties and all tab files if even a single 1 of these things 
            # is missing
            if not np.all(exist):
                print("Loading P, T from velocity model saved in\n", v_path)
                T, P = v_model.T, v_model.P                            
                #for i, f in enumerate(thermo_data.c_field_names[0]):
                for tab,f in zip(thermodata.tabs, thermodata.c_field_names[0]):
                    print("... working on %s ..." % f)
                    thermo_field = ThermoElasticField(tab, f)  
                    thermo_field.extract(T, P, model_name)     
                    thermo_field.save(save_path)
            else:
                print("It looks like everything was already done for",
                      "this model at this time step.")
                
            print("Done")
            print("-"*76)
        print("+"*76)

def compute_vmodels(proj, use_stagyy_rho=False):
    """
    

    Parameters
    ----------
    proj : class Proj
        Your chimera project, from which all relevant information is taken.
    use_stagyy_rho : bool, optional
        Uses stagyy density instead of that obtained via Perple_X. 
        The default is False.

    Returns
    -------
    v_model_paths : TYPE
        DESCRIPTION.

    """
    count = 0
    v_model_paths = []
    zip_names = zip(proj.stagyy_model_names, proj.thermo_data_names)
    for model_name, thermo_name in zip_names:
        thermo_path = proj.thermo_data_path + thermo_name
        parent_path = proj.chimera_project_path + proj.project_name + "/" 
        model_path = parent_path + model_name
        
        print("Loading velocity models obtained from", model_path)
        indices, years = proj.t_indices[model_name], proj.time_span_Gy
        print("using the perplex tables saved as ", thermo_path)
        print("for each of time steps in", years, 
              "Gy, corresponding to indices", indices)
        print()
        
        thermodata = ThermoData.load(thermo_path)
        for i_t, t in zip(indices, years):
            snap_path = model_path + "/{}/".format(i_t)
            v_path = snap_path + proj.vel_model_path
            print("Loading velocity model saved in", v_path)
            v_model = VelocityModel.load(v_path)
            
            moduli_location = snap_path + proj.elastic_path
            print("Averaging rho, K, G")
            v_model.average(*v_model.load_moduli(moduli_location, 
                                                 thermodata.proj_names_dict))
            
            print("Computing seismic velocities")
            v_model.compute_velocities(use_stagyy_rho)
            print("Overwriting velocity model saved in", v_path)
            v_model_paths.append(v_path)
            v_model.save(v_path)
            
            print("Done")
            print("-"*76)
        count += 1
        print("%i/%i models done" % (count, 
                                     len(proj.stagyy_model_names)))
        print("+"*76)        
    return v_model_paths
 
def export_vmodels(proj, fmt="%.18e"):
    """
    

    Parameters
    fmt : str 
        The single format used to save the values with numpy.savetxt 
        Default is ".2f"
    ----------
    proj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
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
            v_model.export(v_path, fmt)
            print("Exporting to sph format")
            print("Done")
            print("----------------------------------------------------------")
            print()
