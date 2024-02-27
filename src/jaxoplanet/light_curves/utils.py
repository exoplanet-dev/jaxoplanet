from functools import wraps
from typing import Any

import jax.numpy as jnp

from jaxoplanet.light_curves.types import LightCurveFunc, Value
from jaxoplanet.types import Array, Quantity


def vectorize(func: LightCurveFunc[Value]) -> LightCurveFunc[Value]:
    @wraps(func)
    def wrapped(time: Quantity, *args: Any, **kwargs: Any) -> Value:
        @jnp.vectorize
        def inner(time_magnitude: Array) -> Value:
            return func(time_magnitude * time.units, *args, **kwargs)

        return inner(time.magnitude)

    return wrapped
