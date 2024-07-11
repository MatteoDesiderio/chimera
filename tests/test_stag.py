"""
TODO: translate to pytest

import unittest
import numpy as np
from stagpy import stagyydata as syyd
from loader import load_coords, load_field


class TestLoader(unittest.TestCase):
    def setUp(self):
        path = "./stagyy_run_test_path/"
        self.stag_path = path
        self.load_coords = load_coords
        self.load_field = load_field

    def test_coords(self):
        sdat = syyd.StagyyData(self.stag_path)
        r, th = self.load_coords(sdat)
        assert isinstance(r, np.ndarray)
        assert isinstance(th, np.ndarray)

    @unittest.skip("Need to rewrite")
    def test_field(self):
        sdat = syyd.StagyyData(self.stag_path)
        f = self.load_field(sdat, "T", -1)
        assert isinstance(f, np.ndarray)
        assert f.shape == (513, 96)  # shape of that particular model is known


if __name__ == "__main__":
    unittest.main()
"""