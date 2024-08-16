"""
generate a list of names for the axisem runs that can be read with by the 
runmanager provided here.
"""

from chimera_project import Project
from numpy import savetxt

# %% provide path of AxiSEM SOLVER and chimera project
SOLVER_path = "/home/matteo/axisem-9f0be2f/SOLVER/"
proj_path = "/home/matteo/chimera-projects/Plum_vs_Marble/"


# %%
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
print("A list has been saved in %s with the name %s" % (SOLVER_path, listname))
print("Please run AxiSEM by executing the run_manager.sh (provided here)")
print("from the SOLVER directory")
print()
print("Usage example: ./run_manager.sh", listname)
print("""
cat %s | while read path run_name
do
# copy the heterogeneities model into the directory
cp $path .
#./submit.csh $run_name
done
""" % listname)
