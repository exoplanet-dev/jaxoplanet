from functools import wraps
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@wraps(jnp.where)
def where(
    condition: ArrayLike,
    x: Optional[ArrayLike],
    y: Optional[ArrayLike],
    *,
    size: Optional[int] = None,
    fill_value: Optional[Union[jax.Array, tuple[ArrayLike]]] = None,
) -> jax.Array:
    """A properly typed version of jnp.where"""
    return jnp.where(condition, x, y, size=size, fill_value=fill_value)  # type: ignore
