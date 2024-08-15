import pytest

from os.path import exists
import numpy as np

from chimera.interfaces.axi.mesh_importer import MeshImporter

def test_importer(tmp_path, example_dir):
    axisem_path = f"{example_dir}/inputData/axisemFolder"
    importer = MeshImporter(axisem_path,
                            mesh_path="PREM_ISO_LIGHT_10s")
    
    temporary_path = tmp_path.as_posix()
    x, y = importer.convert_to_numpy(temporary_path)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # assert x.shape[-1] == 2
    assert exists(f"{temporary_path}/{importer.mesh_name}_x.npy")
    assert exists(f"{temporary_path}/{importer.mesh_name}_y.npy")
