from functools import wraps
from typing import Any, Union

import jax
from jpu.core import is_quantity

from jaxoplanet.light_curves.types import LightCurveFunc
from jaxoplanet.types import Array, Quantity


def vectorize(func: LightCurveFunc) -> LightCurveFunc:
    @wraps(func)
    def wrapped(time: Quantity, *args: Any, **kwargs: Any) -> Union[Array, Quantity]:
        if is_quantity(time):
            time_magnitude = time.magnitude
            time_units = time.units
        else:
            time_magnitude = time
            time_units = None

        def inner(time_magnitude: Array) -> Union[Array, Quantity]:
            if time_units is None:
                return func(time_magnitude, *args, **kwargs)
            else:
                return func(time_magnitude * time_units, *args, **kwargs)

        for _ in time.shape:
            inner = jax.vmap(inner)

        return inner(time_magnitude)

    return wrapped
