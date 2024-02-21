from collections.abc import Callable
from functools import wraps
from typing import Optional, TypeVar

import jax
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg

try:
    from jax.extend import linear_util as lu
except ImportError:
    from jax import linear_util as lu  # type: ignore

T = TypeVar("T", Array, Quantity)


@units.quantity_input(exposure_time=ureg.d)
def integrate_exposure_time(
    func: Callable[[Quantity], T],
    exposure_time: Optional[Quantity] = None,
    order: int = 0,
    num_samples: int = 7,
) -> Callable[[Quantity], T]:
    if exposure_time is None:
        return func

    if jnpu.ndim(exposure_time) != 0:  # type: ignore
        raise ValueError(
            "The exposure time passed to 'integrate_exposure_time' has shape "
            f"{jnpu.shape(exposure_time)}, but a scalar was expected; "  # type: ignore
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
    @units.quantity_input(time=ureg.d)
    def wrapped(time: Quantity) -> T:
        if jnpu.ndim(time) != 0:  # type: ignore
            raise ValueError(
                "The time passed to 'integrate_exposure_time' has shape "
                f"{jnpu.shape(time)}, but a scalar was expected; "  # type: ignore
                "To use exposure time integration for an array of times, "
                "manually 'vmap' or 'vectorize' the function"
            )

        f = lu.wrap_init(jax.vmap(func))
        f = apply_exposure_time_integration(f, stencil, dt)
        return f.call_wrapped(time)  # type: ignore

    return wrapped


@lu.transformation  # type: ignore
def apply_exposure_time_integration(stencil, dt, time):
    result = yield (time + dt,), {}
    yield jnpu.dot(stencil, result)  # type: ignore
