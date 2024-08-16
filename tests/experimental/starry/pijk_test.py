import numpy as np
import pytest

from jaxoplanet.experimental.starry.basis import A1
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.ylm import Ylm


def test_n2ijk():
    assert Pijk.index_to_ijk(5) == (1, 0, 1)
    assert Pijk.index_to_ijk(8) == (0, 2, 0)


@pytest.mark.parametrize("degree", range(15))
def test_pijk_diagonal(degree):
    u = np.ones(degree)
    y = Ylm.from_limb_darkening(u)
    yp = A1(y.deg) @ y.todense()
    p = Pijk.from_dense(yp)

    assert p.diagonal

    if degree > 0:
        yp[Ylm.lm_to_index(degree, -degree)] = 1
        yp[Ylm.lm_to_index(degree, -degree + 1)] = 1
        p = Pijk.from_dense(yp)

        assert not p.diagonal
