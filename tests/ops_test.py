# mypy: ignore-errors

import jax.numpy as jnp
import numpy as np
import pytest
from jax.config import config

from exo4jax import ops
from exo4jax._src.quad import quad_soln_impl

#
# Kepler's equation
#


def get_mean_and_true_anomaly(eccentricity, eccentric_anomaly):
    M = eccentric_anomaly - eccentricity * jnp.sin(eccentric_anomaly)
    f = 2 * jnp.arctan(
        jnp.sqrt((1 + eccentricity) / (1 - eccentricity))
        * jnp.tan(0.5 * eccentric_anomaly)
    )
    return M, f


def check_kepler(e, E):
    atol = 1e-6

    # At single point precision there are numerical issues in both
    # the mod and the trig functions; wrap first
    if e.dtype == jnp.float32:
        E = E % (2 * jnp.pi)
        atol = 1e-5

    M, f = get_mean_and_true_anomaly(e, E)
    sinf0, cosf0 = ops.kepler(M, e)
    np.testing.assert_allclose(sinf0, jnp.sin(f), atol=atol)
    np.testing.assert_allclose(cosf0, jnp.cos(f), atol=atol)


def test_basic_low_e():
    e = jnp.linspace(0, 0.85, 500)
    E = jnp.linspace(-300, 300, 1001)
    e = e[None, :] + jnp.zeros((E.shape[0], e.shape[0]))
    E = E[:, None] + jnp.zeros_like(e)
    check_kepler(e, E)


@pytest.mark.skipif(
    not config.jax_enable_x64,
    reason="Kepler solver has numerical issues at single point precision",
)
def test_basic_high_e():
    e = jnp.linspace(0.85, 1.0, 500)[:-1]
    E = jnp.linspace(-300, 300, 1001)
    e = e[None, :] + jnp.zeros((E.shape[0], e.shape[0]))
    E = E[:, None] + jnp.zeros_like(e)
    check_kepler(e, E)


@pytest.mark.skipif(
    not config.jax_enable_x64,
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


#
# Quadratic limb darkening
#


@pytest.mark.skipif(
    not config.jax_enable_x64,
    reason="Can only compare to exoplanet-core results at double precision",
)
def check_limbdark(b, r, **kwargs):
    from exoplanet_core.jax.ops import quad_solution_vector

    expect = quad_solution_vector(
        b + jnp.zeros_like(r, dtype=jnp.float64),
        r + jnp.zeros_like(b, dtype=jnp.float64),
    )

    calc = quad_soln_impl(b, r, order=10, **kwargs)
    np.testing.assert_allclose(calc, expect, atol=1e-6)


def test_b_grid():
    b = jnp.linspace(-1.5, 1.5, 100_001)
    r = 0.2
    check_limbdark(b, r)


def test_r_grid():
    b = 0.345
    r = jnp.linspace(1e-4, 10.0, 100_001)
    check_limbdark(b, r)


def test_full_grid():
    b = jnp.linspace(-1.5, 1.5, 1001)
    r = jnp.linspace(1e-4, 10.0, 1003)
    b, r = jnp.meshgrid(b, r)
    check_limbdark(b, r)
