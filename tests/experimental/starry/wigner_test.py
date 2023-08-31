import numpy as np
import pytest

from jaxoplanet.experimental.starry.rotation import dotR
from jaxoplanet.experimental.starry.wigner import dot_rotation_matrix
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_matches_dotR(l_max, u):
    theta = 0.1
    np.random.seed(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M = np.random.rand(5, n_max)
    expected = dotR(l_max, u)(M, theta)
    calc = dot_rotation_matrix(l_max, *u, theta)(M)
    assert_allclose(calc, expected)


@pytest.mark.parametrize("l_max", [10, 7, 5, 4])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_compare_starry_rot_rotation(l_max, u):
    """Comparison test with starry OpsYlm.dotR"""
    starry = pytest.importorskip("starry")
    theta = 0.1
    np.random.seed(l_max)
    n_max = l_max**2 + 2 * l_max + 1
    M = np.random.rand(5, n_max)
    m = starry._core.core.OpsYlm(l_max, 0, 0, 1)
    expected = m.dotR(M, *u, theta)
    calc = dot_rotation_matrix(l_max, *u, theta)(M)
    assert_allclose(calc, expected)
