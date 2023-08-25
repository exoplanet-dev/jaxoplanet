import jax.numpy as jnp

from jaxoplanet import orbits


def test_keplerian_central_shape():
    assert orbits.KeplerianCentral.init(mass=0.98, radius=0.93).shape == ()


def test_casting_dtype():
    orbit = orbits.KeplerianBody.init(period=1)
    assert orbit.period.dtype in (jnp.float32, jnp.float64)
