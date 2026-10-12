__all__ = ["light_curve"]

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.core.pizzetti import light_curve as _pizzetti_light_curve
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Scalar


def light_curve(
    orbit: LightCurveOrbit,
    *u: Array,
    order: int = 40,
    method: str = "agol",
    series_tol: float = 1e-10,
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
        method (str): Either ``"agol"`` (the solution of Agol et al. 2020 at every
            time) or ``"pizzetti"`` (an elementary series with a rigorous error
            bound while the transiting body is entirely inside the stellar disk,
            and the former elsewhere); see :mod:`jaxoplanet.core.pizzetti`. The
            latter evaluates all the times at once, one body at a time, and is
            much faster for long, time-ordered arrays of times. If the returned
            function is itself wrapped in ``jax.vmap`` the speed-up is lost.
        series_tol (float): Only for ``method="pizzetti"``: the bound on the
            truncation error of the series relative to the central depth of the
            transit

    Returns:
        A function which takes the time in days as input and returns the light curve flux
    """

    if u:
        ld_u = jnp.concatenate([jnp.atleast_1d(jnp.asarray(u_)) for u_ in u], axis=0)
    else:
        ld_u = jnp.array([])

    if method == "pizzetti":
        return _pizzetti_orbit_light_curve(orbit, ld_u, order, series_tol)
    elif method != "agol":
        raise ValueError(f"Unknown method '{method}'; expected 'agol' or 'pizzetti'")

    @vectorize
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

    return jax.jit(light_curve_impl)


def _pizzetti_orbit_light_curve(
    orbit: LightCurveOrbit, ld_u: Array, order: int, series_tol: float
) -> Callable[[Scalar], Array]:
    """The light curve with the interior series, evaluated on all the times at once

    The series is only worth it if the (few) points near the stellar limb can be
    gathered into a short array, so unlike the default implementation this does not
    ``vmap`` a scalar function over the times. Bodies are processed in sequence
    with ``lax.map``, which keeps the gather per body.
    """

    def light_curve_impl(time: Array) -> Array:
        time = jnp.asarray(time)
        t = time.reshape(-1)

        r_star = orbit.central_radius
        x, y, z = jax.vmap(orbit.relative_position)(t)  # each (len(t), *orbit.shape)

        b = jnp.sqrt(x**2 + y**2) / r_star
        r = orbit.radius / r_star

        shape = jnp.shape(b)[1:]
        nbody = int(np.prod(shape, dtype=int))
        b = b.reshape(t.shape[0], nbody).T
        r = jnp.broadcast_to(r, shape).reshape(nbody)

        def one_body(args: tuple[Array, Array]) -> Array:
            b_, r_ = args
            return _pizzetti_light_curve(
                ld_u, b_, r_, order=order, series_tol=series_tol
            )

        lc = jax.lax.map(one_body, (b, r)).T.reshape(t.shape[0], *shape)
        lc = jnp.where(z > 0, lc, 0)
        return lc.reshape(time.shape + shape)

    return jax.jit(light_curve_impl)
