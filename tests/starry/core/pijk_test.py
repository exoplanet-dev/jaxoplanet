import pytest
from jaxoplanet.starry.core.polynomials import Pijk, polynomial_product_matrix
from jaxoplanet.test_utils import assert_allclose


def test_n2ijk():
    assert Pijk.n2ijk(5) == (1, 0, 1)
    assert Pijk.n2ijk(8) == (0, 2, 0)


def test_additon():
    a = Pijk({(0, 0, 0): 1, (1, 0, 0): 2})
    b = Pijk({(0, 0, 0): 3, (1, 1, 0): 4})
    c = a + b
    assert c.data == {(0, 0, 0): 4, (1, 0, 0): 2, (1, 1, 0): 4}


@pytest.mark.parametrize(
    "p1",
    [
        Pijk.from_dense([0.1, 0.2, 0.3, 0.1]),
        Pijk.from_dense([0.1, -0.2]),
    ],
)
@pytest.mark.parametrize(
    "p2",
    [
        Pijk.from_dense([0.8, 0.3, 0.0, 0.1]),
        Pijk.from_dense([0.5, -0.8]),
    ],
)
def test_product_matrix(p1, p2):
    F1 = polynomial_product_matrix(p1, p2.degree)
    calc = F1 @ p2.todense()
    expected = (p1 * p2).todense()

    assert_allclose(calc, expected)
