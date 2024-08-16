import os

import numpy as np
from stagpy import stagyydata

from .field import Field
from .interfaces.perp.thermo_elastic_field import ThermoElasticField
from .interfaces.stag import loader
from .thermo_data import ThermoData
from .utils import set_renormalized_fields
from .velocity_model import VelocityModel


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

    # load axisem high resolution grid, if wanted
    x, y = proj.get_mesh_xy()

    if proj.quick_mode_on:
        interp_type = "none"
        print("Quick Mode on: overriding interp_type to '%s'." % interp_type)
    else:
        print("Using external mesh.")
        if proj._regular_rect_mesh:
            print("Provided mesh is rectangular.")

    if proj.test_mode_on:
        if checker_board_params is None:
            checker_board_params = [1.0, 0.0, 10, 10, 1600.0, 3100.0]

        print("Test Mode: 'time_span_Gy' and 'stagyy_model_names' ",
              "were used but bear no meaning. Stagyy model is used to load ",
              "coordinates and P, but T field is a checkerboard, ",
              "while the comp field is all pyrolitic")
        for model_name, thermo_name in zip(proj.stagyy_model_names,
                                           proj.thermo_data_names, strict=False):

            print("Initializing variables and compositional fields")
            thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
            variables = [Field(proj, v) for v in thermodata.thermo_var_names]
            fields = [Field(proj, v) for v in thermodata.c_field_names[0]]
            rho_stagyy = Field("rho")
            print("Variables:", *thermodata.thermo_var_names)
            print("Compositional Fields:", *thermodata.c_field_names[0])
            print()

            parent_path = proj.chimera_project_path + proj.project_name + "/"
            model_path = parent_path + model_name
            indices, years = proj.t_indices[model_name], proj.time_span_Gy

            for i_t, t in zip(indices, years, strict=False):
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
                                        proj.c_field_names[0], proj)
                # interpolating stagyy fields on larger axisem grid

                v_model.T = variables[0].interpolate(interp_type, x, y)
                v_model.P = variables[1].interpolate(interp_type, x, y)

                for i, f in enumerate(fields):
                    v_model.C[i] = f.interpolate(interp_type, x, y)

                snap_path = model_path + f"/{i_t}/"
                v_path = snap_path + proj.vel_model_path
                print()
                print("Saving velocity model for", t, "Gy in", v_path )
                v_model.thermo_data_name = proj.thermo_data_path + thermo_name
                v_model.save(v_path)
                print("Done")
                print()

    else:
        zipnames = zip(proj.stagyy_model_names, proj.thermo_data_names, strict=False)
        # load data from stagyy with stagpy into fields
        # TODO speed up the process by loading coords, fields at the same time
        for model_name, thermo_name in zipnames:
            thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
            lims = thermodata.range
            print("Initializing variables and compositional fields")
            variables = [Field(proj, v) for v in thermodata.thermo_var_names]
            fields = [Field(proj, v) for v in thermodata.c_field_names[0]]
            rho_stagyy = Field(proj, "rho")

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
            for i_t, t in zip(indices, years, strict=False):
                print("Loading coords and values with stagpy")
                sdat = stagyydata.StagyyData(proj.stagyy_path + model_name)

                for v, key in zip(variables, lims, strict=False):
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
                                        thermodata.c_field_names[0], proj)
                # interpolating stagyy fields on larger axisem grid
                print("Interpolating stagyy variables",
                      "and fields on axisem mesh")
                v_model.T = variables[0].interpolate(interp_type, x, y)
                v_model.P = variables[1].interpolate(interp_type, x, y)
                v_model.rho_stagyy = rho_stagyy.interpolate(interp_type, x, y)

                for i, f in enumerate(fields):
                    v_model.C[i] = f.interpolate(interp_type, x, y)

                snap_path = model_path + f"/{i_t}/"
                v_path = snap_path + proj.vel_model_path
                print()
                print("Saving velocity model for", t, "Gy in", v_path )
                v_model.thermo_data_name = proj.thermo_data_path + thermo_name
                v_model.save(v_path)
                print("Done")
                print("-" * 50)

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
    zip_names = zip(proj.stagyy_model_names, proj.thermo_data_names, strict=False)
    # in principle, there might be a different thermo data set 4 each model
    if _all_equals(proj.thermo_data_names):
        print("The same thermodynamic dataset was assigned to all models")
        _thermodata = ThermoData.load(proj.thermo_data_path +
                                      proj.thermo_data_names[0])
        _tree = ThermoElasticField.get_tree(_thermodata.tabs[0])
    else:
        _tree, _thermodata = None, None

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

        if _thermodata is _tree is None:
            thermodata = ThermoData.load(proj.thermo_data_path + thermo_name)
            tree = ThermoElasticField.get_tree(thermodata.tabs[0])
        else:
            thermodata = _thermodata
            tree = _tree

        for i_t, _t in zip(indices, years, strict=False):
            snap_path = model_path + f"/{i_t}/"
            v_path = snap_path + proj.vel_model_path
            v_model = VelocityModel.load(v_path)
            save_path = snap_path + proj.elastic_path

            # checking if there is anything missing
            all_exist = _check_all_exist(thermodata, save_path)

            if not all_exist:
                print("Loading P, T from velocity model saved in\n", v_path)
                T, P = v_model.T, v_model.P
                inds = ThermoElasticField.get_indices(tree, T, P)
                for tab,f in zip(thermodata.tabs, thermodata.c_field_names[0], strict=False):
                    print("... working on %s ..." % f)
                    thermo_field = ThermoElasticField(tab, f)
                    thermo_field.extract(inds, model_name)
                    thermo_field.save(save_path)
            else:
                print("It looks like everything was already done for",
                      "this model at time step %i." % i_t)

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
    zip_names = zip(proj.stagyy_model_names, proj.thermo_data_names, strict=False)
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
        for i_t, _t in zip(indices, years, strict=False):
            snap_path = model_path + f"/{i_t}/"
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

def export_vmodels(proj, absolute=True, fac=100, fmt="%.2f", dtype="float32",
                                               fname="geodynamic_hetfile.sph"):
    """

    Parameters
    ----------
    proj : TYPE
        DESCRIPTION.
    absolute : TYPE, optional
        DESCRIPTION. The default is True.
    fac : TYPE, optional
        DESCRIPTION. The default is 100.
    fmt : TYPE, optional
        DESCRIPTION. The default is "%.2f".
    dtype : TYPE, optional
        DESCRIPTION. The default is "float32".
    fname : TYPE, optional
        DESCRIPTION. The default is "geodynamic_hetfile.sph".

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
        for i_t, _t in zip(indices, years, strict=False):
            snap_path = model_path + f"/{i_t}/"
            v_path = snap_path + proj.vel_model_path
            print("Loading velocity model saved in", v_path)
            v_model = VelocityModel.load(v_path)
            print("Exporting to %s format" % fname.rsplit(".")[-1])
            v_model.export(v_path, fmt, absolute, fac, fname, dtype)
            print("Done")
            print("----------------------------------------------------------")
            print()

def _check_all_exist(thermodata, save_path):
    exist = []
    for tab, _f in zip(thermodata.tabs, thermodata.c_field_names[0], strict=False):
        nm = save_path + tab.tab["title"]
        for v in ["rho", "K", "G"]:
            exist.append(os.path.exists(nm + "_" + v + ".npy"))
    # conservatively, we will redo the look-up for all elastic
    # properties and all tab files if even a single 1 of these things
    # is missing
    return np.all(exist)

def _all_equals(x):
    return x.count(x[0]) == len(x)
