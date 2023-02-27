import jax.numpy as jnp
from jaxoplanet import orbits


def test_keplerian_central_shape():
    assert orbits.KeplerianCentral.init(mass=0.98, radius=0.93).shape == ()


def test_casting_dtype():
    orbit = orbits.KeplerianBody.init(period=1)
    assert (
        orbit.period.dtype == jnp.float32 or orbit.period.dtype == jnp.float64
    )
