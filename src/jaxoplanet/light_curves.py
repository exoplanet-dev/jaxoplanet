from functools import partial
from typing import NamedTuple, Optional

import jax.numpy as jnp

from jaxoplanet.core.limb_dark import light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array


class LimbDarkLightCurve(NamedTuple):
    u: Array

    @classmethod
    def init(cls, *u: Array) -> "LimbDarkLightCurve":
        if u:
            u = jnp.concatenate([jnp.atleast_1d(u0) for u0 in u], axis=0)
        else:
            u = jnp.array([])
        return cls(u=u)

    def light_curve(
        self,
        orbit: LightCurveOrbit,
        t: Array,
        *,
        texp: Optional[Array] = None,
        oversample: int = 7,
        order: int = 10,
    ) -> Array:
        """Get the light curve for an orbit at a set of times

        Args:
            orbit: An object with a ``get_relative_position`` method that
                takes a tensor of times and returns a list of Cartesian
                coordinates of a set of bodies relative to the central source.
                The first two coordinate dimensions are treated as being in
                the plane of the sky and the third coordinate is the line of
                sight with positive values pointing *away* from the observer.
                For an example, take a look at :class:`orbits.KeplerianOrbit`.
            t: The times where the light curve should be evaluated.
            texp (Optional[tensor]): The exposure time of each observation.
                This can be a scalar or a tensor with the same shape as ``t``.
                If ``texp`` is provided, ``t`` is assumed to indicate the
                timestamp at the *middle* of an exposure of length ``texp``.
            oversample: The number of function evaluations to use when
                numerically integrating the exposure time.
        """
        del oversample
        if texp is not None:
            raise NotImplementedError(
                "Exposure time integration is not yet implemented"
            )
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(t)
        b = jnp.sqrt(x**2 + y**2)
        r = orbit.radius / r_star
        lc_func = partial(light_curve, self.u, order=order)
        if orbit.shape == ():
            b /= r_star
            lc = lc_func(b, r)
        else:
            b /= r_star[..., None]
            lc = jnp.vectorize(lc_func, signature="(k),()->(k)")(b, r)
        return jnp.where(z > 0, lc, 0)
