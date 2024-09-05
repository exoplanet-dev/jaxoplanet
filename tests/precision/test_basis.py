import numpy as np
import pytest

from jaxoplanet.experimental.starry import basis
from jaxoplanet.experimental.starry.multiprecision import basis as mp_basis, utils

TOLERANCE = 1e-15


@pytest.mark.skip(reason="Not precise enough FOR lmax>10")
@pytest.mark.parametrize("lmax", [15, 10, 7, 5, 4, 3, 2, 1, 0])
def test_A1(lmax):
    expected = np.atleast_2d(utils.to_numpy(mp_basis.A1(lmax)))
    calc = np.atleast_2d(basis.A1(lmax).todense())
    np.testing.assert_allclose(calc, expected, atol=TOLERANCE)


@pytest.mark.parametrize("lmax", [15, 10, 7, 5, 4, 3, 2, 1, 0])
def test_A2_inv(lmax):
    expected = np.atleast_2d(utils.to_numpy(mp_basis.A2_inv(lmax)))
    calc = np.atleast_2d(basis.A2_inv(lmax).todense())
    np.testing.assert_allclose(calc, expected, atol=TOLERANCE)


@pytest.mark.parametrize("lmax", [15, 10, 7, 5, 4, 3, 2, 1, 0])
def test_direct_A2(lmax):
    # count 30s for lmax=20
    expected = np.atleast_2d(utils.to_numpy(mp_basis.A2(lmax)))
    calc = np.atleast_2d(np.linalg.inv(basis.A2_inv(lmax).todense()))
    np.testing.assert_allclose(calc, expected, atol=TOLERANCE)
