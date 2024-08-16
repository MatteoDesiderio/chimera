import numpy as np
from stagpy.stagyydata import StagyyData

from chimera.interfaces.stag.loader import load_coords, load_field


def test_coords(example_dir):
    stag_path = f"{example_dir}/inputData/stagyyModel"
    sdat = StagyyData(stag_path)

    r, th = load_coords(sdat)
    assert isinstance(r, np.ndarray)
    assert isinstance(th, np.ndarray)

def test_field(example_dir):
    stag_path = f"{example_dir}/inputData/stagyyModel"
    sdat = StagyyData(stag_path)

    for var in "T", "p_s", "hz", "bs", "prim":
        f = load_field(sdat, var, -1)
        assert isinstance(f, np.ndarray)
        # assert f.shape == (513, 96)

def test_compositions_almost_one(example_dir):
    stag_path = f"{example_dir}/inputData/stagyyModel"
    sdat = StagyyData(stag_path)

    sum_compositional_fields = 0
    for var in "hz", "bs", "prim":
        f = load_field(sdat, var, -1)
        assert isinstance(f, np.ndarray)
        sum_compositional_fields += f
    difference = np.abs(np.mean(sum_compositional_fields) - 1)
    assert  difference <= 0.01
