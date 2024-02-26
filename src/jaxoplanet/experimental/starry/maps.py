from functools import partial
from typing import Optional

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
    """Surface map object.

    Args:
        y (Optional[Union[Ylm, float]], optional): Ylm object containing the
        spherical harmonic expansion of the map. Defaults to None (a uniform map)
        with amplitude 1.0.
        inc (Optional[float], optional): inclination of the map. Defaults to 0.
        obl (Optional[float], optional): obliquity iof the map. Defaults to 0.
        u (Optional[tuple], optional): polynomial limb-darkening coefficients of
        the map. Defaults to ().
        period (Optional[float], optional): rotation period of the map. Defaults to
        1e15.
        amplitude (Optional[float], optional): amplitude of the map, this quantity is
        proportional to the luminosity of the map and multiplies all flux-related
        observables. Defaults to 1..
        normalize (Optional[bool], optional): whether to normalize the coefficients
        of the spherical harmonics. If None or True, Ylm is normalized and the
        amplitude of the map is set to y[(0, 0)]. Defaults to None.

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
    y: Optional[Ylm] = None

    # Inclination of the map in radians.
    inc: Optional[float] = None

    # Obliquity of the map in radians.
    obl: Optional[float] = None

    # Tuple of limb darkening coefficients.
    u: Optional[tuple] = None

    # Rotation period of the map in days (attribute subject to change)
    period: Optional[float] = None

    # Amplitude of the map, a quantity proportional to map luminosity.
    amplitude: Optional[float] = None

    # Whether to normalize the coefficients of the spherical harmonic.
    normalize: Optional[bool] = None

    def __init__(
        self,
        *,
        y: Optional[Ylm] = None,
        inc: Optional[float] = None,
        obl: Optional[float] = None,
        u: Optional[tuple] = None,
        period: Optional[float] = None,
        amplitude: Optional[float] = None,
        normalize: Optional[bool] = None,
    ):
        if y is None:
            y = Ylm()
        if inc is None:
            inc = jnp.pi / 2
        if obl is None:
            obl = 0.0
        if u is None:
            u = ()
        if period is None:
            period = 1e15
        if amplitude is None:
            amplitude = 1.0
        if normalize is None:
            normalize = True

        if normalize:
            amplitude = y[(0, 0)]
            y = Ylm(data=y.data, normalize=True)

        self.y = y
        self.inc = inc
        self.obl = obl
        self.u = u
        self.period = period
        self.amplitude = amplitude

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

    def flux(self, time):
        theta = time * 2 * jnp.pi / self.period
        return (
            jnp.vectorize(partial(map_light_curve, self, 0.0, 2.0, 2.0, 2.0))(theta)
            * self.amplitude
        )
