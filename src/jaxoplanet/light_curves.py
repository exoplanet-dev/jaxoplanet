from functools import partial
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.limb_dark import light_curve
from jaxoplanet.proto import LightCurveOrbit
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg


class LimbDarkLightCurve(eqx.Module):
    u: Array

    def __init__(self, *u: Array):
        if u:
            self.u = jnp.concatenate([jnp.atleast_1d(u0) for u0 in u], axis=0)
        else:
            self.u = jnp.array([])

    @units.quantity_input(t=ureg.d, texp=ureg.s)
    def light_curve(
        self,
        orbit: LightCurveOrbit,
        t: Quantity,
        *,
        texp: Optional[Quantity] = None,
        oversample: Optional[int] = 7,
        texp_order: Optional[int] = 0,
        limbdark_order: Optional[int] = 10,
    ) -> Array:
        """Get the light curve for an orbit at a set of times

        Args:
            orbit: An object with a ``relative_position`` method that
                takes a tensor of times and returns a list of Cartesian
                coordinates of a set of bodies relative to the central source.
                The first two coordinate dimensions are treated as being in
                the plane of the sky and the third coordinate is the line of
                sight with positive values pointing *away* from the observer.
                For an example, take a look at :class:`orbits.KeplerianOrbit`.
            t: The times where the light curve should be evaluated.
            texp (Optional[Array]): The exposure time of each observation.
                This can be an Array with the same shape as ``t``.
                If ``texp`` is provided, ``t`` is assumed to indicate the
                timestamp at the *middle* of an exposure of length ``texp``.
            oversample: The number of function evaluations to use when
                numerically integrating the exposure time. Defaults to 7.
            texp_order (Optional[int]): The order of the numerical integration
                scheme. This must be one of the following: ``0`` for a
                centered Riemann sum (equivalent to the "resampling" procedure
                suggested by Kipping 2010), ``1`` for the trapezoid rule, or
                ``2`` for Simpson's rule. Has no effect if ``texp`` is not set.
                Defaults to 0.
            limbdark_order (Optional[int]): The quadrature order passed to the
                ``scipy.special.roots_legendre`` function in ``core.limb_dark.py``
                which implements Gauss-Legendre quadrature. Defaults to 10.
        """

        t = jnpu.atleast_1d(t)

        # Handle exposure time integration
        if texp is None:
            tgrid = t
        else:
            if (
                jnp.ndim(texp) == 0
            ):  # If a scalar is passed, automatically broadcast to the shape of t
                texp = jnpu.broadcast_to(texp, jnp.shape(t))
            assert texp.shape == t.shape

            # Ensure oversample is an odd number
            oversample = int(oversample)
            oversample += 1 - oversample % 2
            stencil = jnp.ones(oversample)

            # Construct exposure time integration stencil
            if texp_order == 0:
                dt = jnp.linspace(-0.5, 0.5, 2 * oversample + 1)[1:-1:2]
            elif texp_order == 1:
                dt = jnp.linspace(-0.5, 0.5, oversample)
                stencil = 2 * stencil
                stencil = stencil.at[0].set(1)
                stencil = stencil.at[-1].set(1)
            elif texp_order == 2:
                dt = jnp.linspace(-0.5, 0.5, oversample)
                stencil = stencil.at[1:-1:2].set(4)
                stencil = stencil.at[2:-1:2].set(2)
            else:
                raise ValueError("order must be 0, 1, or 2")
            stencil /= jnp.sum(stencil)

            if jnp.ndim(texp) == 0:  # Unnecessary since I check the shapes above?
                dt = texp * dt
            else:
                dt = jnpu.outer(texp, dt)
            tgrid = (jnpu.reshape(t, newshape=(t.size, 1)) + dt).flatten()

        # Evaluate the coordinates of the transiting body
        x, y, z = orbit.relative_position(tgrid)
        b = jnpu.sqrt(x**2 + y**2)
        r_star = orbit.central_radius
        r = orbit.radius / r_star
        lc_func = partial(light_curve, self.u, order=limbdark_order)
        if orbit.shape == ():
            b /= r_star
            lc: Array = lc_func(b.magnitude, r.magnitude)
        else:
            b /= r_star[..., None]
            lc = jnp.vectorize(lc_func, signature="(k),()->(k)")(
                b.magnitude, r.magnitude
            )
        lc = jnp.where(z > 0, lc, 0)

        # Integrate over exposure time
        if texp is not None:
            if orbit.shape == ():
                lc = jnp.reshape(lc, newshape=(t.size, oversample))
            else:
                lc = jnp.reshape(lc, newshape=(orbit.shape[0], t.size, oversample))
            lc = jnp.dot(lc, stencil)
        return lc
