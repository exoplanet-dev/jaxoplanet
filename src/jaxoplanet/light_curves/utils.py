__all__ = ["vectorize"]

from functools import wraps
from typing import Any

import jax
from jpu.core import is_quantity

from jaxoplanet.light_curves.types import LightCurveFunc
from jaxoplanet.types import Array, Quantity


def vectorize(func: LightCurveFunc) -> LightCurveFunc:
    """Vectorize a scalar light curve function to work with array inputs

    Like ``jax.numpy.vectorize``, this automatically wraps a function which operates on a
    scalar to handle array inputs. Unlike that function, this handles ``Quantity`` inputs
    and outputs, but it only broadcasts the first input (``time``).

    Args:
        func: A function which takes a scalar ``Quantity`` time as the first input

    Returns:
        An updated function which can operate on ``Quantity`` times of any shape
    """

    @wraps(func)
    def wrapped(time: Quantity, *args: Any, **kwargs: Any) -> Array | Quantity:
        if is_quantity(time):
            time_magnitude = time.magnitude
            time_units = time.units
        else:
            time_magnitude = time
            time_units = None

        def inner(time_magnitude: Array) -> Array | Quantity:
            if time_units is None:
                return func(time_magnitude, *args, **kwargs)
            else:
                return func(time_magnitude * time_units, *args, **kwargs)

        for _ in time.shape:
            inner = jax.vmap(inner)

        return inner(time_magnitude)

    return wrapped
