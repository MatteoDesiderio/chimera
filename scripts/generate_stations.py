"""
Created on Tue Jul  5 12:56:36 2022.

@author: matteo
"""
import numpy as np


def generate(n=200, min_max_lat=(-89, 89), min_max_lon=(-179, 179),
             make_grid = False, name = "X", network = "XX",
             out_path = "/home/matteo/axisem-9f0be2f/SOLVER/STATIONS.CUSTOM"):

    min_lat, max_lat = min_max_lat
    min_lon, max_lon = min_max_lon

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
    np.savetxt(out_path, data, fmt="%s")
