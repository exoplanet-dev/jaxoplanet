from functools import partial
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, U0, poly_basis
from jaxoplanet.experimental.starry.light_curves import map_light_curve
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
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

    @partial(jax.jit, static_argnames=("res",))
    def render(self, theta: float = 0.0, res: int = 400):
        _, xyz = ortho_grid(res)
        pT = self.poly_basis(*xyz)
        Ry = left_project(self.ydeg, self.inc, self.obl, theta, 0.0, self.y.todense())
        A1Ry = A1(self.ydeg).todense() @ Ry
        p_y = Pijk.from_dense(A1Ry, degree=self.ydeg)
        U = jnp.array([1, *self.u])
        p_u = Pijk.from_dense(U @ U0(self.udeg), degree=self.udeg)
        p = (p_y * p_u).todense()
        return jnp.reshape(pT @ p, (res, res))

    def flux(self, time):
        theta = time * 2 * jnp.pi / self.period
        return jnp.vectorize(partial(map_light_curve, self, 0.0, 2.0, 2.0, 2.0))(theta)
