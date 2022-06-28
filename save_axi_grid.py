#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:01:44 2022

@author: matteo
"""
from interfaces.axi.mesh_importer import MeshImporter


axisem_path = "/home/matteo/axisem-9f0be2f"
importer = MeshImporter(axisem_path, mesh_path="PREM_ISO_2s")
x, y = importer.convert_to_numpy("/home/matteo/chimera-projects")
