__all__ = ["light_curve"]

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Scalar


def light_curve(
    orbit: LightCurveOrbit, *u: Array, order: int = 10
) -> Callable[[Scalar], Array]:
    """Compute the light curve for arbitrary polynomial limb darkening

    See `Agol et al. (2020) <https://arxiv.org/abs/1908.03222>`_ and
    :func:`jaxoplanet.core.limb_dark.light_curve` for more technical details.

    Args:
        orbit (LightCurveOrbit): An orbit object that can be used to evaluate the
            relative positions of the transiting body with respect to the light source.
        u (Array): The coefficients of the polynomial limb darkening
        order (int): The order of the numerical integration used by the backend; see
            :func:`jaxoplanet.core.limb_dark.light_curve`

    Returns:
        A function which takes the time in days as input and returns the light curve flux
    """

    if u:
        ld_u = jnp.concatenate([jnp.atleast_1d(jnp.asarray(u_)) for u_ in u], axis=0)
    else:
        ld_u = jnp.array([])

    def light_curve_impl(time: Scalar) -> Array:
        if jnp.ndim(time) != 0:
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{jnp.shape(time)}, but a scalar was expected; "
                "this shouldn't typically happen so please open an issue "
                "on GitHub demonstrating the problem"
            )

        # Evaluate the coordinates of the transiting body
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)

        b = jnp.sqrt(x**2 + y**2) / r_star
        r = orbit.radius / r_star

        lc_func = partial(_limb_dark_light_curve, ld_u, order=order)
        lc = lc_func(b, r)
        lc = jnp.where(z > 0, lc, 0)

        return lc

    return jax.vmap(light_curve_impl)
