"""
Simple wrappers to load coordinates and field data via stagpy.
This assumes 2D yz spherical geometry.
"""

from types import MappingProxyType

from stagpy import phyvars
from stagpy._step import Step, _Fields
from stagpy.datatypes import Varf


class CustomStagFields(_Fields):
    def __init__(self, step, variables, extravars, files, filesh5):
        super().__init__(step, variables, extravars, files, filesh5)

        # casting to dict because Mapping Proxy does not allow assignment
        # is this 'dangerous'?
        variables, filesh5 = dict(variables), dict(filesh5)

        # adding our custom mappings
        variables["p_s"] = Varf("Static Pressure", "Pa")
        variables["bs"] = Varf("Basalt fraction", "1")
        variables["hz"] = Varf("Harzburgite fraction", "1")
        filesh5["Pressure"] = ["p_s"]
        filesh5["Basalt"] = ["bs"]
        filesh5["Harzburgite"] = ["hz"]

        # back to normal
        variables = MappingProxyType(variables)
        filesh5 = MappingProxyType(filesh5)

        self._vars = variables
        self._extra = extravars
        self._files = files
        self._filesh5 = filesh5


def load_coords(stagyydata):
    """
    Simple stagpy wrapper to load coordinates from a StagyyData file.

    Parameters
    ----------
    stagyydata : stagpy.stagyydata.StagyyData
        The model data output from stagyy, as read by stagpy.

    Returns
    -------
    r : numpy.ndarray
        radial coordinates.
    theta : numpy.ndarray
        azimuthal coordinates.

    """
    snap = stagyydata.snaps[0]
    r = snap.geom.z_centers
    theta = snap.geom.y_centers
    return (r, theta)


def load_field(stagyydata, name, i):
    """
    Simple stagpy wrapper to load a field from a StagyyData file.

    Parameters
    ----------
    stagyydata : stagpy.stagyydata.StagyyData
        The model data output from stagyy, as read by stagpy.
    name : str
        name of the field.
    i : int
        snapshot index.

    Returns
    -------
    TYPE numpy.ndarray
        2D array representing the field.

    """
    snap = stagyydata.snaps[i]
    snap_copy = Step(snap.istep, snap.sdat)

    snap_copy.fields = CustomStagFields(
        snap,
        phyvars.FIELD,
        phyvars.FIELD_EXTRA,
        phyvars.FIELD_FILES,
        phyvars.FIELD_FILES_H5,
    )

    return snap_copy.fields[name].values.squeeze()  # noqa: PD011
