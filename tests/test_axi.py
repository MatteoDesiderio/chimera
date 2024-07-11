import numpy as np
from chimera.interfaces.axi.mesh_importer import MeshImporter
from os.path import exists



def test_importer(self):
    axisem_path = "./axisem_test_path/"
    importer = MeshImporter(axisem_path, mesh_path="PREM_ISO_10s")

    x, y = importer.convert_to_numpy("./")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # assert x.shape[-1] == 2
    assert exists("./" + self.importer.mesh_name + "_x.npy")
    assert exists("./" + self.importer.mesh_name + "_y.npy")
