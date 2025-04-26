"""A module providing decorators to transform light curve functions"""

__all__ = ["integrate", "interpolate"]

from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp

from jaxoplanet.light_curves.types import LightCurveFunc
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.types import Array, Scalar

try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu  # type: ignore


def integrate(
    func: LightCurveFunc,
    exposure_time: Scalar | None = None,
    order: int = 0,
    num_samples: int = 7,
) -> LightCurveFunc:
    """Transform a light curve function to apply exposure time integration

    This transformation applies a fixed stencil numerical integration scheme to the input
    function ``func`` to convolve the light curve with a top hat exposure time centered
    on the input time, with a full width of ``exposure_time``.

    The order of the integration scheme is set using the ``order`` parameter which must
    be ``0``, ``1``, or ``2``. The default (``0``) uses the "resampling" scheme discussed
    by `Kipping (2010) <https://arxiv.org/abs/1004.3741>`_. The higher order schemes
    ``1`` and ``2`` apply the trapezoid and Simpson's rules respectively, but won't
    necessarily provide higher accuracy results because of discontinuities at the
    contact points.

    In practice, the parameter ``num_samples`` which sets the number of function
    evaluations per integral has the most significant effect on the accuracy of this
    integral, trading off against higher computational cost.

    Args:
        func: A light curve function which takes a time ``Scalar`` as the first
            argument
        exposure_time (Scalar): The exposure time (in days, by default)
        order (int): The order of the integration scheme as discussed above
        num_samples (int): The number of function evaluations made per integral,
            controlling the accuracy of the numerics

    Returns:
        A new light curve function with the same signature as ``func``, computing the
        exposure time integrated flux
    """
    if exposure_time is None:
        return func

    if jnp.ndim(exposure_time) != 0:
        raise ValueError(
            "The exposure time passed to 'integrate_exposure_time' has shape "
            f"{jnp.shape(exposure_time)}, but a scalar was expected; "
            "To use exposure time integration with different exposures at different "
            "times, manually 'vmap' or 'vectorize' the function"
        )

    # Ensure 'num_samples' is an odd number
    num_samples = int(num_samples)
    num_samples += 1 - num_samples % 2
    stencil = jnp.ones(num_samples)

    # Construct exposure time integration stencil
    if order == 0:
        dt = jnp.linspace(-0.5, 0.5, 2 * num_samples + 1)[1:-1:2]
    elif order == 1:
        dt = jnp.linspace(-0.5, 0.5, num_samples)
        stencil = 2 * stencil
        stencil = stencil.at[0].set(1)
        stencil = stencil.at[-1].set(1)
    elif order == 2:
        dt = jnp.linspace(-0.5, 0.5, num_samples)
        stencil = stencil.at[1:-1:2].set(4)
        stencil = stencil.at[2:-1:2].set(2)
    else:
        raise ValueError(
            "The parameter 'order' in 'integrate_exposure_time' must be 0, 1, or 2"
        )
    dt = dt * exposure_time
    stencil /= jnp.sum(stencil)

    @wraps(func)
    @vectorize
    def wrapped(time: Scalar, *args: Any, **kwargs: Any) -> Array | Scalar:
        if jnp.ndim(time) != 0:
            raise ValueError(
                "The time passed to 'integrate_exposure_time' has shape "
                f"{jnp.shape(time)}, but a scalar was expected; "
                "this shouldn't typically happen so please open an issue "
                "on GitHub demonstrating the problem"
            )

        f = lu.wrap_init(jax.vmap(func, in_axes=(0,) + (None,) * len(args)))
        f = apply_exposure_time_integration(f, stencil, dt)  # type: ignore
        return f.call_wrapped(time, args, kwargs)  # type: ignore

    return wrapped


@lu.transformation  # type: ignore
def apply_exposure_time_integration(stencil, dt, time, args, kwargs):
    result = yield (time + dt,) + args, kwargs
    yield jnp.dot(stencil, result)


def interpolate(
    func: LightCurveFunc,
    *,
    period: Scalar,
    time_transit: Scalar,
    num_samples: int,
    duration: Scalar | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
) -> LightCurveFunc:
    """Transform a light curve function to pre-compute the model on a grid

    Sometimes it can be useful to precompute the light curve on a grid near a transit,
    and then interpolate those computations to the required phases when computing the
    full model. This can speed things up a lot when you have many transits, or a lot of
    out of transit data. This transform uses linear interpolation.

    .. note:: Unlike some other transforms, this function requires that any upstream
        ``*args`` and ``**kwargs`` be passed directly to the transform, rather than when
        calling the transformed function. This is necessary because the model is
        pre-computed when it is tranformed.

    Args:
        func: A light curve function which takes a time ``Scalar`` as the first
            argument
        period (Scalar): The period of the orbit. Used to wrap the input times into the
            domain of the pre-computed model
        time_transit (Scalar): The transit time of the orbit. Used to wrap the input
            times into the domain of the pre-computed model
        duration (Scalar): The duration centered on the transit to pre-compute. By
            default, the full period will be evaluated
        num_samples (int): The number of points in the time grid used for pre-computation
        args (tuple): Any extra positional arguments that should be passed to ``func``
        kwargs (dict): Any extra keyword arguments that should be passed to ``func``

    Returns:
        A new light curve function with the same signature as ``func``, computing the
        flux by interpolating a pre-computed model
    """

    kwargs = kwargs or {}
    if duration is None:
        duration = period
    time_grid = time_transit + duration * jnp.linspace(-0.5, 0.5, num_samples)
    flux_grid = func(time_grid, *args, **kwargs)
    flux_magnitude = flux_grid

    @wraps(func)
    @vectorize
    def wrapped(time: Scalar, *args: Any, **kwargs: Any) -> Array | Scalar:
        del args, kwargs
        time_wrapped = (
            jnp.mod(time - time_transit + 0.5 * period, period)
            + 0.5 * period
            + time_transit
        )
        flux = jnp.interp(
            time_wrapped,
            time_grid,
            flux_magnitude,
            left=flux_magnitude[0],
            right=flux_magnitude[-1],
            period=period,
        )
        return flux

    return wrapped
