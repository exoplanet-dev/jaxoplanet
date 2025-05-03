import jax.numpy as jnp
import pytest

from jaxoplanet.experimental import calc_poly_coeffs
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize(
    "coeffs",
    [
        [0.2],
        [0.011, 0.06],
        [0.6, 0.6],
        [0.54, 0.32, 0.2],
    ],
)
@pytest.mark.parametrize("mu_array_size", [100, 1000])
def test_calc_poly_coeff(coeffs, mu_array_size):
    coeffs = jnp.array(coeffs)

    def generate_profile(mu, coeffs):
        x = 1 - mu
        profile = 1
        for order, coeff in enumerate(coeffs):
            profile -= coeff * (x ** (order + 1))
        return profile

    mu = jnp.linspace(0, 1, num=mu_array_size, endpoint=True)
    profile = generate_profile(mu, coeffs)

    calculated_coeffs = calc_poly_coeffs(
        mu,
        profile,
        poly_degree=len(coeffs),
    )

    assert_allclose(coeffs, calculated_coeffs)
