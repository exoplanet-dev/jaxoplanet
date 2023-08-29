import jax
import jax.numpy as jnp
import pytest

from jaxoplanet import orbits
from jaxoplanet.test_utils import assert_allclose, assert_quantity_allclose
from jaxoplanet.units import unit_registry as ureg


def test_keplerian_central_shape():
    assert orbits.KeplerianCentral(mass=0.98, radius=0.93).shape == ()


def test_keplerian_central_density():
    star = orbits.KeplerianCentral()
    assert_quantity_allclose(
        star.density, 1.4 * ureg.g / ureg.cm**3, atol=0.01, convert=True
    )


def test_keplerian_body_keplers_law():
    orbit = orbits.KeplerianBody(semimajor=1.0 * ureg.au)
    assert_quantity_allclose(orbit.period, 1.0 * ureg.year, atol=0.01, convert=True)

    orbit = orbits.KeplerianBody(period=1.0 * ureg.year)
    assert_quantity_allclose(orbit.semimajor, 1.0 * ureg.au, atol=0.01, convert=True)


@pytest.mark.parametrize("prefix", ["", "central_", "relative_"])
@pytest.mark.parametrize(
    "params",
    [
        {
            "central": orbits.KeplerianCentral(mass=1.3, radius=1.1),
            "mass": 0.1,
            "time_transit": 0.1,
            "period": 12.5,
            "inclination": 0.3,
        },
        {
            "central": orbits.KeplerianCentral(mass=1.3, radius=1.1),
            "mass": 0.1,
            "time_transit": 0.1,
            "period": 12.5,
            "inclination": 0.3,
            "eccentricity": 0.3,
            "omega_peri": -1.5,
            "asc_node": 0.3,
        },
    ],
)
def test_keplerian_body_velocity(prefix, params):
    t = jnp.linspace(0.0, 50.0, 500)
    orbit = orbits.KeplerianBody(**params)
    v = getattr(orbit, f"{prefix}velocity")(t)
    for i, v_ in enumerate(v):
        assert_allclose(
            v_.magnitude,
            jax.vmap(
                jax.grad(lambda t: getattr(orbit, f"{prefix}position")(t)[i].magnitude)
            )(t),
        )
