import sys
from glob import glob
from os import mkdir

import numpy as np
from paraview.simple import (
    CreateWriter,
    CreateXYPlotView,
    Delete,
    GetDisplayProperties,
    GetLookupTableForArray,
    GetSources,
    GroupDatasets,
    LegacyVTKReader,
    PlotOverLine,
    SetActiveSource,
    Show,
    UpdatePipeline,
    remove,
)


# %% directories and profiles we're interested in
# where axisem is
def collect_arguments():
    if sys.argv[1] in ("--help", "-h"):
        msg_line_1 = "1st argument) Absolute Path to AxiSEM SOLVER folder\n"
        msg_line_2 = "2nd argument) Name of chimera project\n"
        msg_line_3 = "              (used to identify runList txt file)\n"

        msg = f"{msg_line_1}{msg_line_2}{msg_line_3}"
        print(msg)

        return None

    return sys.argv[1], sys.argv[2]


def run(arguments):
    if arguments:
        axisem_parent_path, project_name = arguments
        # # where your axisem_runS are
        runlist = axisem_parent_path + "runList_" + project_name + ".txt"
        _, run_dirs = np.loadtxt(runlist, dtype=str, unpack=True)

        if not isinstance(run_dirs, np.ndarray):
            run_dirs = [run_dirs]

        # what type of 1D profiles would you like? when iso, vsh = vsv etc.
        fields = ["rho", "vph", "vsh"]

        # if True, will load the data such that they can be visualized
        # as usual (change colorscale etc.)
        fast = True

        # %% geometry of the profiles
        lateral_resolution = 512  # spacing between profiles
        radial_resolution = 511
        thetas = np.linspace(-np.pi / 2, np.pi / 2, lateral_resolution)
        R = 6371e3

        # %% a useful function to visualize the data as normal in paraview

        def make_visible(model, lookupTable):
            SetActiveSource(model)
            representation = Show()
            representation.SelectionPointFieldDataArrayName = "data"
            representation.ColorArrayName = ("POINT_DATA", "data")
            representation.LookupTable = lookupTable
            return representation

        # %% loop over fields
        for run_dir in run_dirs:
            for field in fields:
                # create a path where you want to store your output
                run_path = axisem_parent_path + run_dir
                output_path = run_path + "/" + field + "_post_axisem_read"
                # unless it exists already
                print(output_path)
                try:
                    mkdir(output_path)
                except OSError:
                    pass
                # determine name of vtk files
                vtkname = f"model_{field}_0*.vtk"
                # collect corresponding paths into a list
                paths = glob(run_path + "/Info/" + vtkname)
                if len(paths) == 0:
                    paths = glob(run_path + "/PX/Info/" + vtkname)
                if len(paths) == 0:
                    paths = glob(run_path + "/PZ/Info/" + vtkname)

                # here the paraview stuff begins
                # load from list of paths
                models, representations = [], []
                for path in paths:
                    model = LegacyVTKReader(FileNames=[path])
                    models.append(model)

                    if not (fast):
                        rep = make_visible(model, GetLookupTableForArray("data", 1))
                        representations.append(rep)

                # select all and make a group
                all_proxies = list(GetSources().values())
                group = GroupDatasets(Input=all_proxies)

                # initialize array to store average.
                # (number of rows is + 1 because paraview is weird)
                avg = np.zeros((radial_resolution + 1, 2))
                # create n radial profiles
                for i, th in enumerate(thetas):
                    x2, y2 = R * np.cos(th), R * np.sin(th)

                    # create line
                    plotOverLine = PlotOverLine(group)
                    plotOverLine.Resolution = radial_resolution
                    plotOverLine.Point1 = [0, 0, 0]
                    plotOverLine.Point2 = [x2, y2, 0]

                    UpdatePipeline()

                    # create window to display this data
                    if i == 0:
                        view = CreateXYPlotView()
                        view.ChartTitle = field
                        view.ShowLegend = 0

                    # get a representation of the line data
                    rep = GetDisplayProperties(plotOverLine)
                    # decide what you want on x axis of plot
                    rep.UseIndexForXAxis = 0  # do not use indices
                    rep.XArrayName = "arc_length"
                    # hide all garbage
                    rep.SeriesVisibility = [
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "data",
                        "1",  # data
                        "vtkValidPointMask",
                        "0",  # boh
                        "arc_length",
                        "0",  # dist in m
                        "Points (0)",
                        "0",  # dist in m
                        "Points (1)",
                        "0",  # boh
                        "Points (2)",
                        "0",  # boh
                        "Points (Magnitude)",
                        "0",  # dist in m
                        "vtkOriginalIndices",
                        "0",
                    ]  # indices

                    # write each profile into a csv file
                    args = (lateral_resolution, radial_resolution, i)  # create name
                    fname = "/data_{}_{}_{}.csv".format(*args)
                    writer = CreateWriter(output_path + fname)  # create writer
                    writer.UpdatePipeline()  # and it's written
                    # load the csv again. We need values and r coord
                    data, _, arc, _, _, _ = np.genfromtxt(
                        output_path + fname, delimiter=",", skip_header=1, unpack=True
                    )
                    # add vel values
                    avg[:, 0] += data

                    # we dont need the csv anymore
                    remove(output_path + fname)

                # compute average and save it
                avg /= lateral_resolution
                # store r coordinates
                avg[:, 1] = arc
                np.savetxt(output_path + "/mean_post_axi_read.txt", avg)

                # clean up paraview
                _ = [Delete(proxy) for proxy in GetSources().values()]
                _ = [Delete(rep) for rep in representations]


if __name__ == "__main__":
    run(collect_arguments())
