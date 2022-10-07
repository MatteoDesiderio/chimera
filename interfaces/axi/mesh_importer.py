#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:31:11 2022

@author: matteo
"""
import vtk
import numpy as np
import os


class MeshImporter:
    """ 
    Load, convert and save an AxiSEM mesh as a numpy array. 

    Arguments
    ---------
    axisem_path : str
        The AxiSEM directory
    mesh_path : str
        The name of the mesh stored in the SOLVER directory.
        The default is "PREM_ISO_2s"
    """

    def __init__(self, axisem_path, mesh_path="PREM_ISO_2s", rE_km=6371.0):
        self.axisem_path = axisem_path
        self.mesh_name = mesh_path
        self.x, self.y = self.loader()
        # for now this is set maually. TODO: read this from inparam_mesh
        self.rE_km = rE_km  # radius of the planet

    def loader(self):
        """
        Convert an AxiSEM vtk mesh into a numpy npy file.

        Returns
        -------
        x, y : numpy.ndarray
            Cordinates of the central points of the cells of the AxiSEM mesh. 
        """
        reader = vtk.vtkGenericDataObjectReader()
        _vtkname = "mesh_domaindecomposition.vtk"
        path = self.axisem_path + "/SOLVER/MESHES/" + self.mesh_name + "/"
        os.chdir(path)
        reader.SetFileName(_vtkname)
        reader.Update()
        x, y = np.array(reader.GetOutput().GetPoints().GetData(),
                        dtype="float64")[:, :-1].T
        return x, y

    def convert_to_numpy(self, path, autoname=True, exclude_core=True,
                         r_core_km=3480.5):
        """
        Save the mesh as an array.

        Parameters
        ----------
        path : str
            path to where the points array is saved.
        autoname : bool, optional
            If True the name of the .npy file is the same as the mesh name 
            given by AxiSEM. Else, it can be specified. The default is True.
        exclude_core : bool, optional
            If True the point corresponding to the core are not saved. 
            if True, the radius of the core must be specified via r_core_km.
            The default is True.
        r_core_km : float, optional
            The core radius in km. All points falling inside this radius will
            not be saved. The default is 3480.5

        Returns
        -------
        out : ndarray
            An array object with the coordinates of the central points of the
            AxiSEM mesh. First and second col are x and y coordinates,
            respectively. Note that the AxiSEM grid (corresponding to 
            a half-circle) is mirrored around the axis to form a complete 
            circle. This array is also saved in the specified path.
        """
        if autoname:
            path += "/" + self.mesh_name

        ratio = r_core_km / self.rE_km  # the axi grid is normalized to R_surf
        mantle = np.hypot(self.x, self.y) > ratio
        _x, _y = self.x[mantle], self.y[mantle]
        np.save(path + "_x", _x)
        np.save(path + "_y", _y)
        # mirrored data
        return np.r_[_x, -_x], np.r_[_y, _y]
