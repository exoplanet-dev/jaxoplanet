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


def get_dtype_eps(*args):
    return jnp.finfo(jax.dtypes.result_type(*args)).eps


@jax.custom_jvp
def zero_safe_sqrt(x):
    return jnp.sqrt(x)


@zero_safe_sqrt.defjvp
def zero_safe_sqrt_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = jnp.sqrt(x)
    cond = jnp.less(x, 10 * get_dtype_eps(x))
    denom = jnp.where(cond, jnp.ones_like(x), x)
    tangent_out = 0.5 * x_dot * primal_out / denom
    return primal_out, tangent_out
