import jax.numpy as jnp
import pytest

from jaxoplanet.light_curves.utils import vectorize


@pytest.mark.parametrize("shape", [(), (10,), (10, 3)])
def test_vectorize_scalar(shape):
    @vectorize
    def lc(time):
        assert time.shape == ()
        return time**2

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
