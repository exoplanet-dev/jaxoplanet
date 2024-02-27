from functools import wraps
from typing import Any, Optional, Union

import jax.numpy as jnp
import jpu.numpy as jnpu
from jpu.core import is_quantity

from jaxoplanet import units
from jaxoplanet.light_curves.types import LightCurveFunc
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import unit_registry as ureg


@units.quantity_input(period=ureg.day, time_transit=ureg.day, duration=ureg.day)
def interpolate(
    func: LightCurveFunc,
    period: Quantity,
    time_transit: Quantity,
    num_samples: int,
    duration: Optional[Quantity] = None,
    args: tuple[Any, ...] = (),
    kwargs: Optional[dict[str, Any]] = None,
) -> LightCurveFunc:
    kwargs = kwargs or {}
    if duration is None:
        duration = period
    time_grid = time_transit + duration * jnp.linspace(-0.5, 0.5, num_samples)
    flux_grid = func(time_grid, *args, **kwargs)

    @wraps(func)
    @units.quantity_input(time=ureg.d)
    @vectorize
    def wrapped(time: Quantity, *args: Any, **kwargs: Any) -> Union[Array, Quantity]:
        del args, kwargs
        time_wrapped = (
            jnpu.mod(time - time_transit + 0.5 * period, period)
            + 0.5 * period
            + time_transit
        )
        if is_quantity(flux_grid):
            return (
                jnp.interp(
                    time_wrapped.magnitude,
                    time_grid.magnitude,
                    flux_grid.magnitude,
                    left=flux_grid.magnitude[0],
                    right=flux_grid.magnitude[-1],
                    period=period.magnitude,
                )
                * flux_grid.units
            )
        else:
            return jnp.interp(
                time_wrapped.magnitude,
                time_grid.magnitude,
                flux_grid,
                left=flux_grid[0],
                right=flux_grid[-1],
                period=period.magnitude,
            )

    return wrapped
