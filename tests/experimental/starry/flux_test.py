import pytest
import numpy as np
import jax.numpy as jnp
from jaxoplanet.experimental.starry.flux import flux
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", [2, 5, 10])
@pytest.mark.parametrize("u", [[], [0.1], [0.2, 0.1]])
def test_compare_starry(deg, u):
    starry = pytest.importorskip("starry")

    # map
    np.random.seed(deg)
    y = np.random.randn((deg + 1) ** 2)
    y[0] = 1.0
    y[1:] *= 1e-2
    inc = np.pi / 2

    # occultor
    yo = np.linspace(-3, 3, 1000)
    ro = 0.01
    zo = 2.0
    xo = 0.0

    # starry
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    ms[:, :] = y
    ms[1 : (len(u) + 1)] = u
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jnp.vectorize(
        flux, signature="(),(),(),(),(),(),(),(n),()->()", excluded={0, 9}
    )(deg, 0.0, xo, yo, zo, ro, inc, 0.0, y, u, 0.0)

    assert_allclose(calc, expected)
