from typing import NamedTuple
from functools import partial

import jax.numpy as jnp
import jax

from exo4jax._src.types import Array
from exo4jax._src.orbit_proto import LightCurveOrbit
from exo4jax._src.quad import light_curve


class Quad(NamedTuple):
    u1: Array
    u2: Array

    @classmethod
    def init(cls, u1: Array, u2: Array) -> "Quad":
        return cls(u1=u1, u2=u2)

    def radius_ratio_from_approx_transit_depth(
        self, depth: Array, impact_param: Array
    ) -> Array:
        """Get the radius ratio corresponding to a particular transit depth

        This result will be approximate and it requires ``|b| < 1`` because it
        relies on the small planet approximation.

        Args:
            depth: The approximate transit depth in relative units
            impact_param: The impact parameter

        Returns:
            ror: The radius ratio that approximately corresponds to the depth
            ``depth`` at impact parameter ``impact_param``.
        """
        f0 = 1 - 2 * self.u1 / 6.0 - 2 * self.u2 / 12.0
        arg = 1 - jnp.sqrt(jnp.maximum(0, 1 - impact_param**2))
        f = 1 - self.u1 * arg - self.u2 * arg**2
        factor = f0 / f
        return jnp.sqrt(jnp.maximum(0, depth * factor))

    def light_curve(
        self, orbit: LightCurveOrbit, t: Array, *, order: int = 10
    ) -> Array:
        r_star = orbit.central.radius
        x, y, z = orbit.relative_position(t)
        b = jnp.sqrt(x**2 + y**2) / r_star
        r = orbit.radius / r_star
        lc_func = partial(light_curve, self.u1, self.u2, order=order)
        lc = jax.vmap(lc_func)(b, r)
        return jnp.where(z > 0, lc, 0)
