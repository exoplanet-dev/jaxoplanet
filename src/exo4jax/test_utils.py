import jax
import jax.numpy as jnp
import numpy as np


def assert_allclose(calculated, expected, *args, **kwargs):
    dtype = jnp.result_type(calculated, expected)
    if dtype == jnp.float64:
        kwargs["atol"] = kwargs.get("atol", 2e-5)
    else:
        kwargs["atol"] = kwargs.get("atol", 2e-2)
    np.testing.assert_allclose(calculated, expected, *args, **kwargs)
