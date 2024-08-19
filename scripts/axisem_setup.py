"""
generate a list of names for the axisem runs that can be read with by the
runmanager provided here.
"""
import sys

from numpy import savetxt

from chimera.chimera_project import Project


def collect_arguments():
    if sys.argv[1] in ("--help", "-h"):
        message = ("1st argument)  Absolute path to AxiSEM SOLVER folder\n"
                   "2nd argument)  Absolute path to chimera project\n"
                   )

        print(message)

        return None

    return sys.argv[1:]

def run(SOLVER_path, proj_path):
    proj = Project.load(proj_path)
    vel_model_path = proj.vel_model_path

    thermo_infos = proj.thermo_data_names
    names = proj.stagyy_model_names
    years = proj.time_span_Gy

    lines = []
    for thermo_info, key in zip(thermo_infos, names, strict=False):
        name = key.rstrip("/")
        indices = proj.t_indices[key]
        for year, index in zip(years, indices, strict=False):
            line1 = (proj_path + "/%s/%i" % (name, index) +
                    vel_model_path + "geodynamic_hetfile.sph")
            line2 = "%s-%s-%s-%1.2f-%i" % (proj.bg_model, name, thermo_info,
                                        year, index)
            lines += [line1 + " " + line2]

    listname = proj.project_name + "-runList"
    savetxt(SOLVER_path + listname, lines, fmt="%s")
    print(f"A list has been saved in {SOLVER_path} with the name {listname}")
    print("Please run AxiSEM by executing the run_manager.sh (provided below)")
    print("from the SOLVER directory")
    print()
    print("Usage example: ./run_manager.sh", listname)
    print("\n\nRun Manager Example:\n\n")
    print(f"""
    cat {listname} | while read path run_name
    do
    # copy the heterogeneities model into the directory
    cp $path .
    #./submit.csh $run_name
    done
    """)

if __name__ == "__main__":
    arguments = collect_arguments()
    if arguments:
        run(*arguments)
