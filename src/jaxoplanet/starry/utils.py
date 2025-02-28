from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from jaxoplanet.starry.core.basis import A1, poly_basis
from jaxoplanet.starry.core.rotation import left_project
from jaxoplanet.units import unit_registry as u

RSUN_DAY_TO_M_S = ((1 * u.R_sun / u.day).to(u.m / u.s)).magnitude


@partial(jax.jit, static_argnums=(0))
def ortho_grid(res: int):
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
    z = jnp.sqrt(1.0 - x**2 - y**2)
    y = y + 0.0 * z  # propagate nans
    x = jnp.ravel(x)[None, :]
    y = jnp.ravel(y)[None, :]
    z = jnp.ravel(z)[None, :]
    lat = 0.5 * jnp.pi - jnp.arccos(y)
    lon = jnp.arctan2(x, z)
    return (lat, lon), (x, y, z)


def lon_lat_lines(n: int = 6, pts: int = 100, radius: float = 1.0):
    assert isinstance(n, int) or len(n) == 2

    if isinstance(n, int):
        n = (n, 2 * n)

    n_lat, n_lon = n

    sqrt_radius = radius

    _theta = np.linspace(0, 2 * np.pi, pts)
    _phi = np.linspace(0, np.pi, n_lat + 1)
    lat = np.array(
        [
            (r * np.cos(_theta), r * np.sin(_theta), np.ones_like(_theta) * h)
            for (h, r) in zip(
                sqrt_radius * np.cos(_phi), sqrt_radius * np.sin(_phi), strict=False
            )
        ]
    )

    _theta = np.linspace(0, np.pi, pts // 2)
    _phi = np.linspace(0, 2 * np.pi, n_lon + 1)[0:-1]
    radii = np.sin(_theta)
    lon = np.array(
        [
            (
                sqrt_radius * radii * np.cos(p),
                sqrt_radius * radii * np.sin(p),
                sqrt_radius * np.cos(_theta),
            )
            for p in _phi
        ]
    )

    return lat, lon


def rotation(inc, obl, theta):
    obl = np.array(obl)
    u = [np.cos(obl), np.sin(obl), 0]
    u /= np.linalg.norm(u)
    u *= -(inc - np.pi / 2)

    R = Rotation.from_rotvec(u)
    R *= Rotation.from_rotvec([0, 0, obl])
    R *= Rotation.from_rotvec([np.pi / 2, 0, 0])
    R *= Rotation.from_rotvec([0, 0, -theta])
    return R


def rotate_lines(lines, inc, obl, theta):
    inc = np.array(inc)
    obl = np.array(obl)
    theta = np.array(theta)
    R = rotation(inc, obl, theta)

    rotated_lines = np.array([R.apply(l.T) for l in lines]).T
    rotated_lines = np.swapaxes(rotated_lines.T, -1, 1)

    return rotated_lines


def plot_lines(lines, axis=(0, 1), ax=None, **kwargs):
    import matplotlib.pyplot as plt

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


def graticule(
    inc: float,
    obl: float,
    theta: float = 0.0,
    pts: int = 100,
    white_contour=True,
    radius: float = 1.0,
    n=6,
    **kwargs,
):
    import matplotlib.pyplot as plt

    kwargs.setdefault("c", kwargs.pop("color", "k"))
    kwargs.setdefault("lw", kwargs.pop("linewidth", 1))
    kwargs.setdefault("alpha", 0.3)

    # plot lines
    lat, lon = lon_lat_lines(pts=pts, radius=radius, n=n)
    lat = rotate_lines(lat, inc, obl, theta)
    plot_lines(lat, **kwargs)
    lon = rotate_lines(lon, inc, obl, theta)
    plot_lines(lon, **kwargs)
    theta = np.linspace(0, 2 * np.pi, 2 * pts)

    # contour
    sqrt_radius = radius
    plt.plot(sqrt_radius * np.cos(theta), sqrt_radius * np.sin(theta), **kwargs)
    if white_contour:
        plt.plot(sqrt_radius * np.cos(theta), sqrt_radius * np.sin(theta), c="w", lw=3)


# s2fft have the same but this one is jitabel
def y1d_to_2d(ydeg: int, flm_1d: np.ndarray) -> np.ndarray:
    """1D starry Ylm to 2D s2fft"""
    new_flm = jnp.zeros((ydeg + 1, 2 * ydeg + 1), dtype=flm_1d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[l, m + ydeg].set(flm_1d[i])
            i += 1

    return new_flm


# s2fft have the same but this one is jitabel
def y2d_to_1d(ydeg: int, flm_2d: np.ndarray) -> np.ndarray:
    """2D starry Ylm to 1D s2fft"""
    new_flm = jnp.zeros((ydeg + 1) ** 2, dtype=flm_2d.dtype)
    i = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            new_flm = new_flm.at[i].set(flm_2d[l, m + ydeg])
            i += 1

    return new_flm


def C(l):
    """Complex to real conversion matrix"""
    # See https://doi.org/10.1016/s0166-1280(97)00185-1 (Blanco 1997, Eq. 19)
    A = np.eye(l, l)[:, ::-1]
    B = np.zeros(l)[:, None]
    C = np.diag((-1) ** np.arange(1, l + 1))

    ABC = np.hstack([A, B, C])
    jABC = np.hstack([1j * A, B, -1j * C])[::-1, :]
    one = np.zeros(2 * l + 1)
    one[l] = np.sqrt(2)

    return np.vstack([jABC, one, ABC]) / np.sqrt(2)
