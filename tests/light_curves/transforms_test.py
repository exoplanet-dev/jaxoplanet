import jax.numpy as jnp
import pytest

from jaxoplanet.light_curves import transforms
from jaxoplanet.test_utils import assert_allclose
from jaxoplanet.units import quantity_input, unit_registry as ureg


@quantity_input(time=ureg.day)
def lc_func1(time):
    return jnp.ones_like(time.magnitude)


@quantity_input(time=ureg.day)
def lc_func2(time):
    return jnp.stack([0.5 * time.magnitude + 0.1, -1.5 * time.magnitude + 3.6], axis=-1)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("lc_func", [lc_func1, lc_func2])
def test_integrate_invariant(order, lc_func):
    time = jnp.linspace(0, 10, 50)
    int_func = transforms.integrate(lc_func, exposure_time=0.1, order=order)
    assert_allclose(int_func(time), lc_func(time))
