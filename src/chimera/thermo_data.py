import os
import pickle

import numpy as np

from .interfaces.perp.tab import Tab


class ThermoData:
    def __init__(self):
        """
        Initialize Thermochemical Data.

        Returns
        -------
        None.

        """
        self.perplex_path = ""  # path to tab files
        self.thermo_var_names = []  # as read by stagyy
        # c fields of stagyy + corresponding perplex tables, order must match!
        self.c_field_names = [[], []]
        self.elastic_path = "/elastic-fields/"
        self.description = ""  # title to describe the set of tab files used
        self.tabs = []
        self.proj_names_dict = {}  # a dictionary associating
        self.range = {}

    @property
    def c_field_names(self):
        return self._c_field_names

    @c_field_names.setter
    def c_field_names(self, val):
        self._c_field_names = val
        cstagyy, cperplex = val
        self.proj_names_dict = dict(zip(cstagyy, cperplex, strict=False))

    def import_tab(self):
        """
        When called, this method creates a Tab instance for each of the
        fields, whose names and corresponding tab file title are stored in the
        attribute c_field_names. These tabs are then stored in the attribute
        tabs.
        The overall P, T range is stored in the attributes Tminmax, Pminmax_Pa.

        Returns
        -------
        None.

        """
        self.tab_files = []
        length = len(self.c_field_names[0])
        Tminmax = np.zeros((length, 2))
        Pminmax_Pa = np.zeros((length, 2))

        for i, (_f, tb) in enumerate(zip(*self.c_field_names, strict=False)):
            inpfl = self.perplex_path + tb + ".tab"
            tab = Tab(inpfl)
            tab.load()
            tab.remove_nans()
            self.tabs.append(tab)
            Tminmax[i] = tab.data[0].min(), tab.data[0].max()
            Pminmax_Pa[i] = tab.data[1].min(), tab.data[1].max()

        Pminmax_Pa *= 1e5
        Tminmax = Tminmax[:, 0].min(), Tminmax[:, 1].max()
        Pminmax_Pa = Pminmax_Pa[:, 0].min(), Pminmax_Pa[:, 1].max()
        self.range[self.thermo_var_names[0]] = Tminmax
        self.range[self.thermo_var_names[1]] = Pminmax_Pa

    def save(self, save_path):
        """
        Save a thermo_data somewhere.

        Parameters
        ----------
        save_path : Str
            Path where the class is saved.

        Returns
        -------
        None.

        """
        parent = save_path + "/ThermoData/"
        try:
            os.mkdir(parent)
        except FileExistsError:
            print(
                "Path "
                + parent
                + " exists. If file "
                + self.description
                + " already exists, it will be overwritten."
            )

        with open(parent + self.description + ".pkl", "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path + ".pkl", "rb") as f:
            return pickle.load(f)
