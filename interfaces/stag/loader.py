"""
Simple wrappers to load coordinates and field data via stagpy. 
This assumes 2D yz spherical geometry.
"""


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
    r = snap.geom.z_coord
    theta = snap.geom.y_coord
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
    return stagyydata.snaps[i].fields[name].squeeze()
