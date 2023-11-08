import matplotlib.pyplot as plt
import numpy as np


def graticule(
    inc=0.0, obl=0.0, n=(8, 6), res=100, axis=(1, 2), ax=None, behind=False, **kwargs
):
    """Plot latitude and longitude lines on a sphere.

    Parameters
    ----------
    inc : float, optional
        inclination, by default 0
    obl : float, optional
        obliquity, by default 0
    n : int or tuple, optional
        the number of latitudes and longitudes. If int interpreted as (n, n),
        by default (8, 6)
    res : int, optional
        resolution of the lines, by default 100
    axis : tuple, optional
        the axes on which to project the line (x:0, y:1, z:2), by default (1, 2)
    ax : matplotlib.axes.Axes, optional
        the axes on which to plot, by default None
    behind : bool, optional
        whether to show lines behind the sphere, by default False. If True front
        lines are hidden.
    """
    kwargs.setdefault("color", kwargs.pop("c", "k"))
    kwargs.setdefault("lw", "1")

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    if isinstance(n, int):
        n = (n, 2 * n)

    n_lat, n_lon = n
    theta = np.linspace(0, 2 * np.pi, res)
    phi = np.linspace(0, np.pi, n_lat + 1)
    latitude_lines = np.array(
        [
            (r * np.cos(theta), r * np.sin(theta), np.ones_like(theta) * h)
            for (h, r) in zip(np.cos(phi), np.sin(phi))
        ]
    )

    theta = np.linspace(0, np.pi, res // 2)
    phi = np.linspace(0, 2 * np.pi, n_lon + 1)[0:-1]
    radii = np.sin(theta)
    longitude_lines = np.array(
        [(radii * np.cos(p), radii * np.sin(p), np.cos(theta)) for p in phi]
    )

    def plot_lines(xyzs, axis=(1, 2), **kwargs):
        # hide lines behind
        other_axis = list(set(axis).symmetric_difference([0, 1, 2]))[0]
        _behind = xyzs[:, other_axis, :] < 0
        if behind:
            _behind = np.logical_not(_behind)
        _xyzs = xyzs.copy().swapaxes(1, 2)
        _xyzs[_behind, :] = np.nan
        _xyzs = _xyzs.swapaxes(1, 2)

        for i, j in _xyzs[:, axis, :]:
            ax.plot(i, j, **kwargs)

    def rotate(xyzs, inc, obl):
        def Ry(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def Rx(theta):
            c, s = np.cos(theta), np.sin(theta)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        rotated_xyzs = np.array([np.dot(Rx(-inc), p) for p in xyzs])
        return np.array([np.dot(Ry(-obl), p) for p in rotated_xyzs])

    plot_lines(rotate(latitude_lines, np.pi / 2 - inc, obl), axis, **kwargs)
    plot_lines(rotate(longitude_lines, np.pi / 2 - inc, obl), axis, **kwargs)

    if not behind:
        phi = np.linspace(0, 2 * np.pi, 3 * res)
        circle = np.array([np.cos(phi), np.sin(phi)])
        ax.plot(*circle, **kwargs)
