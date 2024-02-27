from functools import partial
from typing import Callable

import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg


def limb_dark_light_curve(
    orbit: LightCurveOrbit, *u: Array, order: int = 10
) -> Callable[[Quantity], Array]:
    if u:
        ld_u = jnp.concatenate([jnp.atleast_1d(jnp.asarray(u_)) for u_ in u], axis=0)
    else:
        ld_u = jnp.array([])

    @units.quantity_input(time=ureg.d)
    def light_curve_impl(time: Quantity) -> Array:
        if jnpu.ndim(time) != 0:
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{jnpu.shape(time)}, but a scalar was expected; "
                "To compute a light curve for an array of times, "
                "manually 'vmap' or 'vectorize' the function"
            )

        # Evaluate the coordinates of the transiting body
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)

        b = jnpu.sqrt(x**2 + y**2) / r_star
        assert b.units == ureg.dimensionless
        r = orbit.radius / r_star
        assert r.units == ureg.dimensionless

        lc_func = partial(_limb_dark_light_curve, ld_u, order=order)
        lc = lc_func(b.magnitude, r.magnitude)
        lc = jnp.where(z > 0, lc, 0)

        return lc

    return light_curve_impl
