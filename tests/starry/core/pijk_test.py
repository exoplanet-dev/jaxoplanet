from jaxoplanet.starry.core.polynomials import Pijk


def test_n2ijk():
    assert Pijk.n2ijk(5) == (1, 0, 1)
    assert Pijk.n2ijk(8) == (0, 2, 0)
