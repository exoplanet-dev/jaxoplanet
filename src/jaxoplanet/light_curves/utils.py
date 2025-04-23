__all__ = ["vectorize"]

from functools import wraps
from typing import Any

import jax

from jaxoplanet.light_curves.types import LightCurveFunc
from jaxoplanet.types import Array, Scalar


def vectorize(func: LightCurveFunc) -> LightCurveFunc:
    """Vectorize a scalar light curve function to work with array inputs

    Like ``jax.numpy.vectorize``, this automatically wraps a function which operates on a
    scalar to handle array inputs. Unlike that function, this handles ``Scalar`` inputs
    and outputs, but it only broadcasts the first input (``time``).

    Args:
        func: A function which takes a scalar ``Scalar`` time as the first input

    Returns:
        An updated function which can operate on ``Scalar`` times of any shape
    """

    @wraps(func)
    def wrapped(time: Scalar, *args: Any, **kwargs: Any) -> Array | Scalar:

        def inner(time_magnitude: Array) -> Array | Scalar:
            return func(time_magnitude, *args, **kwargs)

        for _ in time.shape:
            inner = jax.vmap(inner)

        return inner(time)

    return wrapped
