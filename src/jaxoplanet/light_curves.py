from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.limb_dark import light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg


class LimbDarkLightCurve(eqx.Module):
    """A light curve"""

    u: Array

    def __init__(self, *u: Array):
        if u:
            self.u = jnp.concatenate([jnp.atleast_1d(u0) for u0 in u], axis=0)
        else:
            self.u = jnp.array([])

    @units.quantity_input(time=ureg.d)
    def __call__(
        self,
        orbit: LightCurveOrbit,
        time: Quantity,
        *,
        order: int = 10,
    ) -> Array:
        """Get the light curve for an orbit at a set of times

        Args:
            orbit: An object with a ``relative_position`` method that
                takes a tensor of times and returns a list of Cartesian
                coordinates of a set of bodies relative to the central source.
                The first two coordinate dimensions are treated as being in
                the plane of the sky and the third coordinate is the line of
                sight with positive values pointing *away* from the observer.
                For an example, take a look at :class:`orbits.keplerian`.
            time: The time where the light curve should be evaluated.
            order: The quadrature order passed to the
                ``scipy.special.roots_legendre`` function in ``core.limb_dark.py``
                which implements Gauss-Legendre quadrature. Defaults to 10.
        """

        if jnpu.ndim(time) != 0:  # type: ignore
            raise ValueError(
                "The time passed to 'light_curve' has shape "
                f"{jnpu.shape(time)}, but a scalar was expected; "  # type: ignore
                "To use exposure time integration for an array of times, "
                "manually 'vmap' or 'vectorize' the function"
            )

        # Evaluate the coordinates of the transiting body
        r_star = orbit.central_radius
        x, y, z = orbit.relative_position(time)
        b = jnpu.sqrt(x**2 + y**2) / r_star  # type: ignore
        r = orbit.radius / r_star
        lc_func = partial(light_curve, self.u, order=order)
        lc: Array = lc_func(b.magnitude, r.magnitude)
        lc = jnp.where(z > 0, lc, 0)

        return lc

    def light_curve(
        self, orbit: LightCurveOrbit, time: Quantity, *, order: int = 10
    ) -> Array:
        return self(orbit, time, order=order)
