from paraview.simple import *
import sys
import numpy as np
from os import mkdir, remove


# %% directories and profiles we're interested in
# where axisem is
axisem_parent_path = '/home/matteo/axisem-9f0be2f/SOLVER/'
project_name = 'Plum_vs_Marble'
# # where your axisem_runS are
runlist = axisem_parent_path + project_name + '-runList'
_, run_dirs = np.loadtxt(runlist, dtype=str, unpack=True)
# what type of 1D profiles would you like? when iso, vsh = vsv etc.
fields = ["rho", "vph", "vsh"]
# if True, will load the data such that they can be visualized
# as usual (change colorscale etc.)
fast = True

# %% geometry of the profiles
lateral_resolution = 3  # spacing between profiles
radial_resolution = 511
thetas = np.linspace(-np.pi/2, np.pi/2, lateral_resolution)
R = 6371e3

# %% a useful function to visualize the data as normal in paraview


def make_visible(model, lookupTable):
    SetActiveSource(model)
    representation = Show()
    representation.SelectionPointFieldDataArrayName = 'data'
    representation.ColorArrayName = ('POINT_DATA', 'data')
    representation.LookupTable = lookupTable
    return representation


# %% loop over fields
for run_dir in run_dirs:
    for field in fields:
        # create a path where you want to store your output
        run_path = axisem_parent_path + run_dir
        output_path = run_path + '/' + field + "_post_axisem_read"
        # unless it exists already
        try:
            mkdir(output_path)
        except OSError:
            pass
        # determine name of vtk files
        vtkname = 'model_{}_0*.vtk'.format(field)
        # collect corresponding paths into a list
        paths = glob(run_path + '/axisem_run/Info/' + vtkname)

        # here the paraview stuff begins
        # load from list of paths
        models, representations = [], []
        for path in paths:
            model = LegacyVTKReader(FileNames=[path])
            models.append(model)

            if not(fast):
                rep = make_visible(model, GetLookupTableForArray("data", 1))
                representations.append(rep)

        # select all and make a group
        all_proxies = GetSources().values()
        group = GroupDatasets(Input=all_proxies)

        # initialize array to store average.
        # (number of rows is + 1 because paraview is weird)
        avg = np.zeros((radial_resolution + 1, 2))
        # create n radial profiles
        for i, th in enumerate(thetas):

            x2, y2 = R*np.cos(th), R*np.sin(th)

            # create line
            plotOverLine = PlotOverLine(group)
            plotOverLine.Source.Resolution = radial_resolution
            plotOverLine.Source.Point1 = [0, 0, 0]
            plotOverLine.Source.Point2 = [x2, y2, 0]

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
            rep.XArrayName = 'arc_length'
            # hide all garbage
            rep.SeriesVisibility = ['', '', '', '', '', '', '', '',
                              'data', '1',                  # data
                              'vtkValidPointMask', '0',     # boh
                              'arc_length', '0',            # dist in m
                              'Points (0)', '0',            # dist in m
                              'Points (1)', '0',            # boh
                              'Points (2)', '0',            # boh
                              'Points (Magnitude)', '0',    # dist in m
                              'vtkOriginalIndices', '0']  # indices

            # write each profile into a csv file
            args = (lateral_resolution, radial_resolution, i)  # create name
            fname = "/data_{}_{}_{}.csv".format(*args)
            writer = CreateWriter(output_path + fname)  # create writer
            writer.UpdatePipeline()                    # and it's written
            # load the csv again. We need values and r coord
            data, _, arc, _, _ = np.genfromtxt(output_path + fname,
                                         delimiter=',',
                                         skip_header=1, unpack=True)
            # add vel values
            avg[:, 0] += data

            # we dont need the csv anymore
            # remove(output_path + fname)

        # compute average and save it
        avg /= lateral_resolution
        # store r coordinates
        avg[:, 1] = arc
        np.savetxt(output_path + "/mean_post_axi_read.txt", avg)

        # clean up paraview
        _ = [Delete(proxy) for proxy in GetSources().values()]
        _ = [Delete(rep) for rep in representations]

        # I thoght the following was necessary but it seems unimportant
        #for i, mod in enumerate(models):
        #   try:
        #       Delete(mod)
        #   except RuntimeError:
        #       print("error", str(i))
