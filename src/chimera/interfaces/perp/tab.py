import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import ma


def _fix_i_field(i_field):
    if i_field < 2:
        raise Warning("The specified index pointed to the independent "
                      "variables. The index has been automatically "
                      "to the minimum acceptable value")
    i_field = max(2, i_field)
    return i_field

def _interp_nans(arr, x, y):
    nr, nc = arr.shape
    # define square around x, y coordinates in array
    dist = 1
    x1, x2 = max(0, x - dist), min(nr, x + 1 + dist)
    y1, y2 = max(0, y - dist), min(nc, y + 1 + dist)
    square = np.array(arr[x1: x2, y1:y2])

    # if all elements in this square are NaNs, make it larger
    while np.all(np.isnan(square)):
        dist += 1
        x1, x2 = max(0, x - dist), min(nr, x + 1 + dist)
        y1, y2 = max(0, y - dist), min(nc, y + 1 + dist)
        square = np.array(arr[x1: x2, y1:y2])

    square = ma.masked_array(square, mask=np.isnan(square))
    return square.mean()


class Tab:
    def __init__(self, inpfl):
        self.inpfl = inpfl
        self.tab = {}
        with open(inpfl) as fl:
            _ = fl.readline().strip()  # skip line..
            title = fl.readline().strip()  # remove blanks...
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            dT = float(fl.readline())
            nT = float(fl.readline())
            _ = fl.readline().strip()
            _ = fl.readline().strip()
            dP = float(fl.readline())
            nP = float(fl.readline())
            nvar = float(fl.readline())
            variables = fl.readline()

        nT, nP, nvar = int(nT), int(nP), int(nvar)

        # read actual data
        print("\nReading table:  " + title)

        fields = variables.split()
        self.tab["title"] = "".join(title.split(".")[:-1])
        self.tab["dT"] = dT
        self.tab["dP"] = dP
        self.tab["nT"] = nT
        self.tab["nP"] = nP
        self.tab["nvar"] = nvar
        self.tab["fields"] = fields

        print("dT: ", dT, ", dP: ", dP, ", nT: ",
              nT, ", nP: ", nP, ", nvar: ", nvar)
        self.data = []  # could parallelize method, but need 2 initialize data

    def load(self):
        nT = self.tab["nT"]
        nP = self.tab["nP"]
        for i, field in enumerate(self.tab["fields"]):
            data_ = np.loadtxt(self.inpfl, skiprows=13, usecols=i)
            if field == "T(K)":
                data_ = data_[:nT]

            elif field == "P(bar)":
                data_ = data_[::nT]

            else:
                data_ = np.reshape(data_, (nP, nT)).T

            self.data.append(data_)

    def save(self):
        for i, field in enumerate(self.tab["fields"]):
            data_ = self.data[i]
            print("\nSaving table to: " + self.tab["title"] + ".npy\n")
            template = "table_{}_{}.npy"
            np.save(template.format(self.tab["title"], field), data_)

        fname = "stats_{}.pkl".format(self.tab["title"])
        with open(fname, "wb") as f:
            pickle.dump(self.tab, f)


    def plot(self, i_field, ax=None, exclude_range=None, kwargs=None):

        if kwargs is None:
            kwargs = {}
        i_field = _fix_i_field(i_field)

        if ax is not None:
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(1)

        fld_name = self.tab["fields"][i_field]
        data = self.data[i_field]
        T, P = self.data[:2]
        # plotting
        arg_given = exclude_range is not None
        is_modulus = "Ks" in fld_name or "Gs" in fld_name
        if arg_given and is_modulus:
            Tmin, Tmax = exclude_range[0]
            Pmin, Pmax = exclude_range[1]
            TT, PP = np.meshgrid(T, P)
            i1 = (Tmin <= TT) & (Tmax >= TT)
            i2 = (Pmin <= PP) & (Pmax >= PP)
            data_ininterval = ma.masked_array(data, i1.T & i2.T)
            vmin, vmax = data_ininterval.min(), data_ininterval.max()
        else:
            vmin, vmax = None, None

        if "vmin" not in kwargs.keys():
            kwargs["vmin"] = vmin
        if "vmax" not in kwargs.keys():
            kwargs["vmax"] = vmax
        img = ax.pcolormesh(T, P, data.T, **kwargs)
        ax.set_title(self.tab["title"], loc="left")
        ax.set_title(self.tab["fields"][i_field], loc="right")
        ax.set_ylabel(self.tab["fields"][1])
        ax.set_xlabel(self.tab["fields"][0])
        plt.colorbar(img, ax=ax)

        return fig, ax

    def get_contour_TP(self, i_field, T, P):
        """


        Parameters
        ----------
        i_field : TYPE
            DESCRIPTION.
        T : TYPE
            DESCRIPTION.
        P : TYPE
            In Pa.

        Returns
        -------
        profile : TYPE
            DESCRIPTION.

        """
        i_field = _fix_i_field(i_field)
        data = self.data[i_field]
        Tax, Pax = self.data[:2]
        profile = np.zeros_like(T)

        for i, (T_, P_) in enumerate(zip(T, P, strict=False)):
            iT = np.argmin(np.abs(T_ - Tax))
            iP = np.argmin(np.abs(P_ - Pax*1e5))
            profile[i] = data[iT, iP]

        return profile

    def plot_contour_TP(self ):
        pass

    def remove_nans(self):
        for k, (f, n) in enumerate(zip(self.data, self.tab["fields"], strict=False)):
            if n != "T(K)" and n != "P(bar)":
                new = np.array(f)
                for where in np.argwhere(np.isnan(f)):
                    i, j = where
                    new[i, j] = _interp_nans(f, i, j)
                self.data[k] = new
