import unittest
import numpy as np
from mesh_importer import MeshImporter
from os.path import exists


class TestFields(unittest.TestCase):
    def setUp(self):
        axisem_path = "/home/matteo/axisem-9f0be2f"
        self.importer = MeshImporter(axisem_path, mesh_path="PREM_ISO_10s")

    def test_importer(self):
        x, y = self.importer.convert_to_numpy("./")
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        # assert x.shape[-1] == 2
        assert exists("./" + self.importer.mesh_name + "_x.npy")
        assert exists("./" + self.importer.mesh_name + "_y.npy")


if __name__ == "__main__":
    unittest.main()
