"""
Created on Tue Jul  5 12:56:36 2022.

@author: matteo
"""

import sys

import numpy as np


def collect_arguments():
    if sys.argv[1] in ("--help", "-h"):
        message = (
            "1st argument)  Absolute Path to AxiSEM-run folder\n"
            "2nd argument)  num. of stations\n"
            "3rd argument)  min. Lat\n"
            "4th argument)  max. Lat\n"
            "5th argument)  min. Lon\n"
            "6th argument)  max. Lon\n"
            "7th argument)  Make Grid (True | False)\n"
            "8th argument)  Station Name\n"
            "9th argument)  Network Name\n"
        )

        print(message)

        return None

    return sys.argv[1:]


def generate(
    out_path,
    n=200,
    min_lat=-89,
    max_lat=89,
    min_lon=-179,
    max_lon=179,
    make_grid=False,
    name="X",
    network="XX",
):
    central_lon = (min_lon + max_lon) / 2.0

    lat_list = np.linspace(min_lat, max_lat, n)

    if make_grid:
        lon_list = np.linspace(min_lon, max_lon, n)

        lons, lats = np.meshgrid(lon_list, lat_list)

        lon_list = lons.flatten()
        lat_list = lats.flatten()
    else:
        lon_list = np.ones(n) * central_lon

    names = [name + f"{k}" for k in range(len(lon_list))]
    nets = [network for k in range(len(lon_list))]

    x = [f"{lat:.3f}" for lat in lat_list]
    y = [f"{lat:.3f}" for lat in lon_list]

    z = ["0.0" for k in range(len(lon_list))]

    data = np.c_[names, nets, x, y, z, z]
    np.savetxt(f"{out_path}/STATIONS.CUSTOM", data, fmt="%s")


if __name__ == "__main__":
    arguments = collect_arguments()
    if arguments:
        generate(*arguments)
