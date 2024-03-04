# mypy: ignore-errors

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from jaxoplanet.core import kepler
from jaxoplanet.test_utils import assert_allclose


def get_mean_and_true_anomaly(eccentricity, eccentric_anomaly):
    M = eccentric_anomaly - eccentricity * jnp.sin(eccentric_anomaly)
    f = 2 * jnp.arctan(
        jnp.sqrt((1 + eccentricity) / (1 - eccentricity))
        * jnp.tan(0.5 * eccentric_anomaly)
    )
    return M, f


def check_kepler(e, E):
    # TODO: Add check for gradients
    E = E % (2 * jnp.pi)
    M, f = get_mean_and_true_anomaly(e, E)
    sinf0, cosf0 = kepler(M, e)
    assert_allclose(sinf0, jnp.sin(f))
    assert_allclose(cosf0, jnp.cos(f))


def test_basic_low_e():
    e = jnp.linspace(0, 0.85, 500)
    E = jnp.linspace(-300, 300, 1001)
    e = e[None, :] + jnp.zeros((E.shape[0], e.shape[0]))
    E = E[:, None] + jnp.zeros_like(e)
    check_kepler(e, E)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Kepler solver has numerical issues at single point precision",
)
def test_basic_high_e():
    e = jnp.linspace(0.85, 1.0, 500)[:-1]
    E = jnp.linspace(-300, 300, 1001)
    e = e[None, :] + jnp.zeros((E.shape[0], e.shape[0]))
    E = E[:, None] + jnp.zeros_like(e)
    check_kepler(e, E)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="Kepler solver has numerical issues at single point precision",
)
def test_edge_cases():
    E = jnp.array([0.0, 2 * jnp.pi, -226.2, -170.4])
    e = (1 - 1e-6) * jnp.ones_like(E)
    e = jnp.append(e[:-1], 0.9939879759519037)
    check_kepler(e, E)


def test_pi():
    e = jnp.linspace(0, 1.0, 100)[:-1]
    E = jnp.pi + jnp.zeros_like(e)
    check_kepler(e, E)


@pytest.mark.parametrize("e", [0.01, 0.1, 0.5, 0.6])
@pytest.mark.xfail(
    not jax.config.jax_enable_x64,
    reason="Gradients can be noisy at single point precision",
)
def test_grad(e):
    M = jnp.linspace(-5, 5, 100)
    check_grads(kepler, (M, e), order=1)
