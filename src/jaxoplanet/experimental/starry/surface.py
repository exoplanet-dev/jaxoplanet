from collections.abc import Iterable
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from jaxoplanet.experimental.starry.basis import A1, U0, poly_basis
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import (
    left_project,
    right_project_axis_angle,
)
from jaxoplanet.experimental.starry.utils import ortho_grid
from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.types import Array


class Surface(eqx.Module):
    """Surface map object.

    Args:
        y: Ylm object containing the spherical harmonic expansion of the map. Defaults to
            a uniform map with amplitude 1.0.
        inc: inclination of the map relative to line of sight.
            Defaults to 90 degrees (pi/2 radians).
        obl: obliquity of the map.
        u: polynomial limb-darkening coefficients of the map.
        period: rotation period of the map.
        amplitude: amplitude of the map; this quantity is proportional to the luminosity
            of the map and multiplies all flux-related observables.
        normalize: whether to normalize the coefficients of the spherical harmonics. If
            True, Ylm is normalized and the amplitude of the map is set to y[(0, 0)].

    Example:

        .. code-block:: python

            import numpy as np
            import jax
            from jaxoplanet.experimental.starry.utils import show_map
            from jaxoplanet.experimental.starry.maps import Map
            from jaxoplanet.experimental.starry.ylm import Ylm

            jax.config.update("jax_enable_x64", True)

            np.random.seed(30)
            y = Ylm.from_dense(np.random.rand(20))
            m = Map(y=y, u=[0.5, 0.1], inc=0.9, obl=-0.3)
            show_map(m)
    """

    # Ylm object representing the spherical harmonic expansion of the map.
    y: Ylm

    # Inclination of the map in radians.
    inc: Array

    # Obliquity of the map in radians.
    obl: Array

    # Tuple of limb darkening coefficients.
    u: tuple[Array, ...]

    # Rotation period of the map in days (attribute subject to change)
    period: Optional[Array]

    # Amplitude of the map, a quantity proportional to map luminosity.
    amplitude: Array

    # Boolean to specify whether the Ylm coefficients should be normalized
    normalize: bool

    def __init__(
        self,
        *,
        y: Optional[Ylm] = None,
        inc: Array = 0.5 * jnp.pi,
        obl: Array = 0.0,
        u: Iterable[Array] = (),
        period: Optional[Array] = None,
        amplitude: Array = 1.0,
        normalize: bool = True,
    ):

        if y is None:
            y = Ylm()

        if normalize:
            amplitude = jnp.array(y[(0, 0)], float)
            y = Ylm(data=y.data).normalize()

        self.y = y
        self.inc = inc
        self.obl = obl
        self.u = tuple(u)
        self.period = period
        self.amplitude = amplitude
        self.normalize = normalize

    @property
    def poly_basis(self):
        return jax.jit(poly_basis(self.deg))

    @property
    def udeg(self):
        return len(self.u)

    @property
    def ydeg(self):
        return self.y.deg

    @property
    def deg(self):
        return self.ydeg + self.udeg

    def _intensity(self, x, y, z, theta=0.0):
        pT = self.poly_basis(x, y, z)
        Ry = left_project(self.ydeg, self.inc, self.obl, theta, 0.0, self.y.todense())
        A1Ry = A1(self.ydeg).todense() @ Ry
        p_y = Pijk.from_dense(A1Ry, degree=self.ydeg)
        U = jnp.array([1, *self.u])
        p_u = Pijk.from_dense(U @ U0(self.udeg), degree=self.udeg)
        p = (p_y * p_u).todense()
        return pT @ p * self.amplitude

    @partial(jax.jit, static_argnames=("res",))
    def render(self, theta: float = 0.0, res: int = 400):
        """Returns the intensity map projected onto the x-y plane (sky).

        Args:
            theta (float, optional): rotation angle of the map. Defaults to 0.0.
            res (int, optional): resolution of the render. Defaults to 400.

        Returns:
            ArrayLike: square 2D array representing the intensity map
            (with nans outside the map disk).
        """
        _, xyz = ortho_grid(res)
        intensity = self._intensity(*xyz, theta=theta)
        return jnp.reshape(intensity, (res, res))

    def intensity(self, lat: float, lon: float):
        """Returns the intensity of the map at a given latitude and longitude.

        Args:
            lat (float): latitude in the rest frame of the map
            lon (float): longitude in the rest frame of the map

        Returns:
            float: intensity of the map at the given latitude and longitude
        """
        lon = lon + jnp.pi / 2  # convention, 0 lon faces the observer
        lat = jnp.pi / 2 - lat  # convention, latitude 0 is equator
        x = jnp.sin(lat) * jnp.cos(lon)
        y = jnp.sin(lat) * jnp.sin(lon)
        z = jnp.cos(lat) * jnp.ones_like(x)

        axis = right_project_axis_angle(self.inc - jnp.pi / 2, self.obl, 0.0, 0.0)
        axis = jnp.array(axis[0:3]) * axis[-1]
        rotation = Rotation.from_rotvec(axis)
        x, y, z = rotation.apply(jnp.array([x, y, z]).T).T

        return self._intensity(x, y, z)

    def rotational_phase(self, time: Array) -> Array:
        if self.period is None:
            return jnp.zeros_like(time)
        else:
            return 2 * jnp.pi * time / self.period
