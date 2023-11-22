import numpy as np
import matplotlib.pyplot as plt
from jaxoplanet.experimental.starry.rotation import right_project_axis_angle
from scipy.spatial.transform import Rotation


def lon_lat_lines(n=6, pts=100):
    if isinstance(n, int):
        n = (n, 2 * n)

    n_lat, n_lon = n

    _theta = np.linspace(0, 2 * np.pi, pts)
    _phi = np.linspace(0, np.pi, n_lat + 1)
    lat = np.array(
        [
            (r * np.cos(_theta), r * np.sin(_theta), np.ones_like(_theta) * h)
            for (h, r) in zip(np.cos(_phi), np.sin(_phi))
        ]
    )

    _theta = np.linspace(0, np.pi, pts // 2)
    _phi = np.linspace(0, 2 * np.pi, n_lon + 1)[0:-1]
    radii = np.sin(_theta)
    lon = np.array(
        [(radii * np.cos(p), radii * np.sin(p), np.cos(_theta)) for p in _phi]
    )

    return lat, lon


def rotation(inc, obl, theta):
    u = [np.cos(obl), np.sin(obl), 0]
    u /= np.linalg.norm(u)
    u *= -(inc - np.pi / 2)

    R = Rotation.from_rotvec(u)
    R *= Rotation.from_rotvec([0, 0, obl])
    R *= Rotation.from_rotvec([np.pi / 2, 0, 0])
    R *= Rotation.from_rotvec([0, 0, theta])
    return R


def rotate_lines(lines, inc, obl, theta):
    R = rotation(inc, obl, theta)

    rotated_lines = np.array([R.apply(l.T) for l in lines]).T
    rotated_lines = np.swapaxes(rotated_lines.T, -1, 1)

    return rotated_lines


def plot_lines(lines, axis=(0, 1), ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    # hide lines behind
    other_axis = list(set(axis).symmetric_difference([0, 1, 2]))[0]
    behind = lines[:, other_axis, :] < 0
    _xyzs = lines.copy().swapaxes(1, 2)
    _xyzs[behind, :] = np.nan
    _xyzs = _xyzs.swapaxes(1, 2)

    for i, j in _xyzs[:, axis, :]:
        ax.plot(i, j, **kwargs)
