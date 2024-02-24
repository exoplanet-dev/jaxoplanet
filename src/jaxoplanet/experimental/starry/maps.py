from functools import partial
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from jaxoplanet.experimental.starry.basis import A1, U0, poly_basis
from jaxoplanet.experimental.starry.light_curves import map_light_curve
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import (
    left_project,
    right_project_axis_angle,
)
from jaxoplanet.experimental.starry.utils import ortho_grid
from jaxoplanet.experimental.starry.ylm import Ylm


class Map(eqx.Module):
    y: Optional[Ylm] = None
    """Ylm object representing the spherical harmonic expansion
    of the map."""
    inc: Optional[float] = None
    """Inclination of the map in radians."""
    obl: Optional[float] = None
    """Obliquity of the map in radians."""
    u: Optional[tuple] = None
    """Tuple of limb darkening coefficients."""
    period: Optional[float] = None
    """Rotation period of the map in days (attribute subject
    to change)"""

    """
    An object containing the spherical harmonic expansion of the map, and its
    orientation in space.

    Example:

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

    def __init__(
        self,
        *,
        y: Optional[Union[Ylm, float]] = None,
        inc: Optional[float] = None,
        obl: Optional[float] = None,
        u: Optional[tuple] = None,
        period: Optional[float] = None,
    ):

        if y is None:
            y = 1.0

        if isinstance(y, float):
            y = Ylm({(0, 0): y, (1, 0): 0.0})

        if inc is None:
            inc = jnp.pi / 2
        if obl is None:
            obl = 0.0
        if u is None:
            u = ()
        if period is None:
            period = 1e15

        self.y = y
        self.inc = inc
        self.obl = obl
        self.u = u
        self.period = period

    @property
    def poly_basis(self):
        return jax.jit(poly_basis(self.deg))

    @property
    def udeg(self):
        return len(self.u)

    @property
    def ydeg(self):
        return self.y.ell_max

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
        return pT @ p

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

    @partial(jax.jit)
    def intensity(self, lat: float, lon: float):
        """Returns the intensity of the map at a given latitude and longitude.

        Args:
            lat (float): latitude in the rest frame of the map
            lon (float): longitude in the rest frame of the map

        Returns:
            float: intensity of the map at the given latitude and longitude
        """
        x = jnp.sin(lat) * jnp.cos(lon)
        y = jnp.sin(lat) * jnp.sin(lon)
        z = jnp.cos(lat) * jnp.ones_like(x)

        axis = right_project_axis_angle(self.inc - jnp.pi / 2, self.obl, 0.0, 0.0)
        axis = jnp.array(axis[0:3]) * axis[-1]
        rotation = Rotation.from_rotvec(axis)
        x, y, z = rotation.apply(jnp.array([x, y, z]).T).T

        return self._intensity(x, y, z)

    def flux(self, theta):
        return jnp.vectorize(partial(map_light_curve, self, None, None, None, None))(
            theta
        )
