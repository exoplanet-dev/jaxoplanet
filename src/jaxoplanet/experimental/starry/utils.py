import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from jaxoplanet.experimental.starry.basis import A1, poly_basis
from jaxoplanet.experimental.starry.rotation import left_project


def ortho_grid(res):
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
    z = jnp.sqrt(1 - x**2 - y**2)
    y = y + 0.0 * z  # propagate nans
    x = jnp.ravel(x)[None, :]
    y = jnp.ravel(y)[None, :]
    z = jnp.ravel(z)[None, :]
    lat = 0.5 * jnp.pi - jnp.arccos(y)
    lon = jnp.arctan2(x, z)
    return (lat, lon), (x, y, z)


def render(deg, res, theta, inc, obl, y):
    _, xyz = ortho_grid(res)
    pT = poly_basis(deg)(*xyz)
    Ry = left_project(deg, inc, obl, theta, 0.0, y)
    A1Ry = A1(deg) @ Ry
    return jnp.reshape(pT @ A1Ry, (res, res))


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
    R *= Rotation.from_rotvec([0, 0, -theta])
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
