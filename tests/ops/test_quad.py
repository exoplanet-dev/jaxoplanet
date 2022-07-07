# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np

from exo4jax._src.quad import quad_soln_impl


def check_limbdark(b, r, **kwargs):
    with jax.experimental.enable_x64():
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
