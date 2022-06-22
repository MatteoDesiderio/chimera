import unittest
import numpy as np
from field import Field
from stagpy import stagyydata as syyd
import matplotlib.pyplot as plt
from interfaces.stag import loader as stag_loader
from utils import to_polar


class TestFields(unittest.TestCase):
    def setUp(self):
        path = "/home/matteo/stagyyRuns/ghostresults/BS_drhoLM350_z1_PrLM100/"
        self.field = Field()
        self.stag_path = path
        self.to_polar = to_polar

    def test_coords(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.coords = stag_loader.load_coords(sdat)
        assert isinstance(self.field.coords[0], np.ndarray)
        assert isinstance(self.field.coords[1], np.ndarray)
        assert len(self.field.coords[0]) != len(self.field.coords[1])
        assert len(self.field.coords[0]) == 96
        x, y = self.field.to_cartesian()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert self.field.r_max is None

    def test_field(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.values = stag_loader.load_field(sdat, self.field.name, -1)
        assert isinstance(self.field.values, np.ndarray)
        assert self.field.values.shape == (513, 96)

    def test_split(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.coords = stag_loader.load_coords(sdat)
        self.field.values = stag_loader.load_field(sdat, self.field.name, -1)
        perc = 0.25
        left, right = self.field.split(perc)
        whole = self.field
        assert isinstance(right, Field)
        assert isinstance(left, Field)
        # implement something similar to this
        # assert len(left.values)*(1+perc) == 513 * (1+perc) / 2
        # assert len(right.values)*(1+perc) == 513 * (1+perc) / 2

        fig, axs = plt.subplots(3, 1)
        fields = [whole, right, left]
        for i in range(len(axs)):
            ax = axs[i]
            ax.imshow(fields[i].values.T, vmin=2500,
                      vmax=3200, cmap="bwr")
            ax.set_xlim(0, 513)
            if i == 0:
                n = 513 // 2
                ax.axvline(perc * n)
                ax.axvline(n - perc * n)
                ax.axvline(n, color="k")
                ax.axvline(n + perc * n)
                ax.axvline(n * 2 - perc * n)

    def test_plot(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.coords = stag_loader.load_coords(sdat)
        self.field.values = stag_loader.load_field(sdat, self.field.name, -1)

        fig, ax = self.field.plot()
        ax.set_title("Stagpy original grid")
        assert ax.has_data()

    def test_interpolate(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.coords = stag_loader.load_coords(sdat)
        self.field.values = stag_loader.load_field(sdat, self.field.name, -1)
        x = np.load("./PREM_ISO_10s_x.npy")
        y = np.load("./PREM_ISO_10s_y.npy")
        znew = self.field.interpolate(x, y, 4,
                                      {"leafsize": 10},
                                      {"r": 0.08, "return_sorted": True})

        assert isinstance(znew, np.ndarray)
        assert znew.size == len(x)
        assert self.field.r_max == 6360047.726103055

        fig = plt.figure()
        ax = fig.gca()
        ax.set_title("Interpolated points")
        ax.tricontourf(x, y, znew, levels=512)
        #ax.scatter(x, y, s=1, c=znew)

    def test_to_polar(self):
        sdat = syyd.StagyyData(self.stag_path)
        self.field.coords = stag_loader.load_coords(sdat)
        self.field.values = stag_loader.load_field(sdat, self.field.name, -1)
        x = np.load("./PREM_ISO_10s_x.npy")
        y = np.load("./PREM_ISO_10s_y.npy")
        znew = self.field.interpolate(x, y, 4,
                                      {"leafsize": 10},
                                      {"r": 0.08, "return_sorted": True})
        r, theta = self.to_polar(x, y)

        r *= self.field.r_max
        theta -= np.pi / 2
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_theta_offset(np.pi / 2.0)
        # ax.set_rmin(3480)
        ax.scatter(theta, r, s=1, c=znew, marker=".")
        #ax.tricontourf(theta, r, znew, levels=512)


if __name__ == "__main__":
    unittest.main()
