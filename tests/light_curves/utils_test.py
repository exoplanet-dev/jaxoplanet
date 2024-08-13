import jax.numpy as jnp
import pytest

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.units import quantity_input, unit_registry as ureg


@pytest.mark.parametrize("shape", [(), (10,), (10, 3)])
def test_vectorize_scalar(shape):
    @quantity_input(time=ureg.day)
    @vectorize
    def lc(time):
        assert time.shape == ()
        return time.magnitude**2

    time = jnp.ones(shape)
    assert lc(time).shape == time.shape


@pytest.mark.parametrize("shape", [(), (10,), (10, 3)])
def test_vectorize_array(shape):
    @vectorize
    def lc(time):
        assert time.shape == ()
        return jnp.array([time, time**2, 1.0])

    time = jnp.ones(shape)
    assert lc(time).shape == time.shape + (3,)


@pytest.mark.parametrize("shape", [(), (10,), (10, 3)])
@pytest.mark.parametrize("out_units", [None, ureg.dimensionless, ureg.m])
def test_vectorize_quantity(shape, out_units):
    @quantity_input(time=ureg.day)
    @vectorize
    def lc(time, units):
        assert time.shape == ()
        y = jnp.stack([time.magnitude, time.magnitude**2, 1.0])
        if units is None:
            return y
        else:
            return y * units

    time = jnp.ones(shape)
    assert lc(time, out_units).shape == time.shape + (3,)
    assert lc(time, units=out_units).shape == time.shape + (3,)
