__all__ = ["light_curve"]

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.power_two import light_curve as _power_two_light_curve
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg


def light_curve(
    orbit: LightCurveOrbit, alpha: Array, c: Array, eps: float = 1e-12
) -> Callable[[Quantity], Array]:
    @units.quantity_input(time=ureg.d)
    @vectorize
    def light_curve_impl(time: Quantity) -> Array:
        if jnpu.ndim(time) != 0:
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{jnpu.shape(time)}, but a scalar was expected; "
                "this shouldn't typically happen so please open an issue "
                "on GitHub demonstrating the problem"
            )

        # Evaluate the coordinates of the transiting body
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)

        b = jnpu.sqrt(x**2 + y**2) / r_star
        assert b.units == ureg.dimensionless
        r = orbit.radius / r_star
        assert r.units == ureg.dimensionless

        lc_func = partial(_power_two_light_curve, alpha, c, eps=eps)
        lc = lc_func(b.magnitude, r.magnitude)
        lc = jnp.where(z > 0, lc, 0)

        return lc

    return light_curve_impl
