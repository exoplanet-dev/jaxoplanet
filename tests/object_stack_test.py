import jax.numpy as jnp
import pytest

from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.test_utils import assert_quantity_pytree_allclose


def f1(x):
    return x["a"] ** 2


def f2(x):
    return {"y": x["a"] ** 2}


@pytest.mark.parametrize(
    "func, objects, result",
    [
        (
            f1,
            (
                {"a": 0.5},
                {"a": 0.1},
            ),
            jnp.asarray([0.5**2, 0.1**2]),
        ),
        (
            f1,
            (
                {"a": 0.5},
                {"a": 0.1, "b": 1000.0},
            ),
            jnp.asarray([0.5**2, 0.1**2]),
        ),
        (
            f2,
            (
                {"a": 0.5},
                {"a": 0.1},
            ),
            {"y": jnp.asarray([0.5**2, 0.1**2])},
        ),
        (
            f2,
            (
                {"a": 0.5},
                {"a": 0.1, "b": 1000.0},
            ),
            {"y": jnp.asarray([0.5**2, 0.1**2])},
        ),
    ],
)
def test_object_stack(func, objects, result):
    stack = ObjectStack(*objects)
    assert_quantity_pytree_allclose(stack.vmap(func)(), result)
