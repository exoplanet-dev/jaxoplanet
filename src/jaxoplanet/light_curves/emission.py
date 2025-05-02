from collections.abc import Callable
from functools import partial

import jax.numpy as jnp

from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.types import Array, Scalar


def light_curve(system: SurfaceSystem, order: int = 10) -> Callable[[Scalar], Array]:

    assert isinstance(
        system, SurfaceSystem
    ), f"Expected an instance of 'SurfaceSystem', but got {type(system)}"

    @partial(jnp.vectorize, signature="()->(n)")
    def light_curve_impl(time: Scalar) -> Array:
        if jnp.ndim(time) != 0:
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{jnp.shape(time)}, but a scalar was expected; "
                "this shouldn't typically happen so please open an issue "
                "on GitHub demonstrating the problem"
            )

        # Evaluate the coordinates of the transiting body
        r_star = system.central.radius
        x, y, z = system.relative_position(time)

        b = jnp.sqrt(x**2 + y**2) / r_star
        r = system.radius / r_star

        lc_func = partial(_limb_dark_light_curve, system.central_surface.u, order=order)
        lc = lc_func(b, r)
        lc = jnp.where(z > 0, lc, 0.0).sum(0) + 1

        bodies_lc = []
        for body, surface in zip(system.bodies, system.body_surfaces, strict=False):
            x, y, z = body.relative_position(time)
            b = jnp.sqrt(x**2 + y**2) / body.radius
            r = r_star / body.radius

            lc_func = partial(_limb_dark_light_curve, surface.u, order=order)
            lc_body = lc_func(b, r)
            lc_body = surface.amplitude * (1 + jnp.where(z < 0, lc_body, 0))

            bodies_lc.append(lc_body)

        return jnp.hstack([lc, *bodies_lc])

    return light_curve_impl
