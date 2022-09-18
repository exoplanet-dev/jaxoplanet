from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from exo4jax._src.proto import LightCurveOrbit
from exo4jax._src.quad import light_curve
from exo4jax._src.types import Array


class QuadLightCurve(NamedTuple):
    """A quadratically limb darkened light curve

    Args:
        u1: The first limb darkening coefficient
        u2: The second limb darkening coefficient
    """

    u1: Array
    u2: Array

    @classmethod
    def init(cls, u1: Array, u2: Array) -> "QuadLightCurve":
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
        self,
        orbit: LightCurveOrbit,
        t: Array,
        *,
        texp: Optional[Array] = None,
        oversample: int = 7,
        order: int = 10
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
            order: The order of the numerical integration scheme used to
                compute the limb darkening integrals.
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
        lc_func = partial(light_curve, self.u1, self.u2, order=order)
        if orbit.shape == ():
            b /= r_star
            lc = lc_func(b, r)
        else:
            b /= r_star[..., None]
            lc = jnp.vectorize(lc_func, signature="(k),()->(k)")(b, r)
        return jnp.where(z > 0, lc, 0)
